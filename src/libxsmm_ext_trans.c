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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_trans.h"
#include "libxsmm_ext.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define LIBXSMM_MCOPY_MT(MT, NT, M, N) ((MT) <= (M) && (NT) <= (N) && \
  (((unsigned int)(LIBXSMM_AVG_M)) * LIBXSMM_AVG_N) <= (((unsigned int)(M)) * (N)))


LIBXSMM_APIEXT void libxsmm_matcopy_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch)
{
#if defined(LIBXSMM_TRANS_CHECK)
  if (0 != out && out != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && m <= ldo)
#endif
  {
    LIBXSMM_INIT
    {
#if defined(_OPENMP)
      const unsigned int tm = libxsmm_trans_mtile[4 < typesize ? 0 : 1];
      const unsigned int tn = (unsigned int)(libxsmm_trans_tile_stretch * tm);
      if (LIBXSMM_MCOPY_MT(tm, tn, (unsigned int)m, (unsigned int)n)) { /* consider problem-size */
        const int iprefetch = (0 == prefetch ? 0 : *prefetch);
        libxsmm_xmcopyfunction kernel = NULL;
        const libxsmm_mcopy_descriptor* desc;
        libxsmm_descriptor_blob blob;
        if (0 != (1 & libxsmm_trans_jit) /* JIT'ted matrix-copy permitted? */
          && NULL != (desc = libxsmm_mcopy_descriptor_init(&blob, typesize,
          tm, tn, (unsigned int)ldo, (unsigned int)ldi,
            0 != in ? 0 : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE, iprefetch, NULL/*default unroll*/)))
        {
          kernel = libxsmm_dispatch_mcopy(desc);
        }

# if defined(LIBXSMM_EXT_TASKS) && 0/* implies _OPENMP */
        if (0 == omp_get_active_level())
# else
        if (0 == omp_in_parallel())
# endif
        { /* enable internal parallelization */
          const int nthreads = omp_get_max_threads();
# if defined(LIBXSMM_EXT_TASKS)
          if (0 >= libxsmm_trans_taskscale)
# endif
          {
#           pragma omp parallel num_threads(nthreads)
            libxsmm_matcopy_thread_internal(out, in, typesize,
              (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
              prefetch, tm, tn, kernel, omp_get_thread_num(), nthreads);
          }
# if defined(LIBXSMM_EXT_TASKS)
          else { /* tasks requested */
            const int ntasks = nthreads * libxsmm_trans_taskscale;
#           pragma omp parallel num_threads(nthreads)
            { /* first thread discovering work will launch all tasks */
#             pragma omp single nowait /* anyone is good */
              { int tid;
                for (tid = 0; tid < ntasks; ++tid) {
#                 pragma omp task untied
                  libxsmm_matcopy_thread_internal(out, in, typesize,
                    (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
                    prefetch, tm, tn, kernel, tid, ntasks);
                }
              }
            }
          }
# endif
        }
        else { /* assume external parallelization */
          const int nthreads = omp_get_num_threads();
# if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
          const int ntasks = (0 == libxsmm_trans_taskscale
            ? (LIBXSMM_TRANS_TASKSCALE)
            : libxsmm_trans_taskscale) * nthreads;
          int tid;
          for (tid = 0; tid < ntasks; ++tid) {
#           pragma omp task untied
            libxsmm_matcopy_thread_internal(out, in, typesize,
              (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
              prefetch, tm, tn, kernel, tid, ntasks);
          }
          if (0 == libxsmm_nosync) { /* allow to omit synchronization */
#           pragma omp taskwait
          }
# else
          libxsmm_matcopy_thread_internal(out, in, typesize,
            (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
            prefetch, tm, tn, kernel, omp_get_thread_num(), nthreads);
# endif
        }
      }
      else
#else
      LIBXSMM_UNUSED(prefetch);
#endif /*defined(_OPENMP)*/
      { /* no MT, or small problem-size */
        LIBXSMM_XCOPY_NONJIT(LIBXSMM_MCOPY_KERNEL,
          typesize, out, in, ldi, ldo, 0, m, 0, n,
          LIBXSMM_XALIGN_MCOPY);
      }
    }
  }
#if defined(LIBXSMM_TRANS_CHECK)
  else {
    static int error_once = 0;
    if (0 != libxsmm_get_verbosity() /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (0 == out) {
        fprintf(stderr, "LIBXSMM ERROR: the matcopy input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXSMM ERROR: output and input of the matcopy must be different!\n");
      }
      else if (0 == typesize) {
        fprintf(stderr, "LIBXSMM ERROR: the typesize of the matcopy is zero!\n");
      }
      else if (0 >= m || 0 >= n) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix extent(s) of the matcopy is/are zero or negative!\n");
      }
      else {
        LIBXSMM_ASSERT(ldi < m || ldo < n);
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the matcopy is/are too small!\n");
      }
    }
  }
#endif
}


LIBXSMM_APIEXT void libxsmm_otrans_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  static int error_once = 0;
#if defined(LIBXSMM_TRANS_CHECK)
  if (0 != out && 0 != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && n <= ldo)
#endif
  {
    LIBXSMM_INIT
    if (out != in) {
#if defined(_OPENMP)
      const unsigned int tm = libxsmm_trans_mtile[4 < typesize ? 0 : 1];
      const unsigned int tn = (unsigned int)(libxsmm_trans_tile_stretch * tm);
      if (0 == LIBXSMM_TRANS_NO_BYPASS(m, n) && tm <= (unsigned int)m && tn <= (unsigned int)n) { /* consider problem-size */
# if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
        if (0 == omp_get_active_level())
# else
        if (0 == omp_in_parallel())
# endif
        { /* enable internal parallelization */
          const int nthreads = omp_get_max_threads();
# if defined(LIBXSMM_EXT_TASKS)
          if (0 >= libxsmm_trans_taskscale)
# endif
          {
#           pragma omp parallel num_threads(nthreads)
            libxsmm_otrans_thread_internal(out, in, typesize,
              (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
              tm, tn, NULL/*kernel*/, omp_get_thread_num(), nthreads);
          }
# if defined(LIBXSMM_EXT_TASKS)
          else { /* tasks requested */
            const int ntasks = nthreads * libxsmm_trans_taskscale;
#           pragma omp parallel num_threads(nthreads)
            { /* first thread discovering work will launch all tasks */
#             pragma omp single nowait /* anyone is good */
              { int tid;
                for (tid = 0; tid < ntasks; ++tid) {
#                 pragma omp task untied
                  libxsmm_otrans_thread_internal(out, in, typesize,
                    (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
                    tm, tn, NULL/*kernel*/, tid, ntasks);
                }
              }
            }
          }
# endif
        }
        else { /* assume external parallelization */
          const int nthreads = omp_get_num_threads();
# if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
          const int ntasks = (0 == libxsmm_trans_taskscale
            ? (LIBXSMM_TRANS_TASKSCALE)
            : libxsmm_trans_taskscale) * nthreads;
          int tid;
          for (tid = 0; tid < ntasks; ++tid) {
#           pragma omp task untied
            libxsmm_otrans_thread_internal(out, in, typesize,
              (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
              tm, tn, NULL/*kernel*/, tid, ntasks);
          }
          if (0 == libxsmm_nosync) { /* allow to omit synchronization */
#           pragma omp taskwait
          }
# else
          libxsmm_otrans_thread_internal(out, in, typesize,
            (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
            tm, tn, NULL/*kernel*/, omp_get_thread_num(), nthreads);
# endif
        }
      }
      else
#endif /*defined(_OPENMP)*/
      { /* no MT, or small problem-size */
        libxsmm_xtransfunction kernel = NULL;
        const libxsmm_trans_descriptor* desc;
        libxsmm_descriptor_blob blob;
        if (0 != (2 & libxsmm_trans_jit) /* JIT'ted transpose permitted? */
          && NULL != (desc = libxsmm_trans_descriptor_init(&blob, typesize, (unsigned int)m, (unsigned int)n, (unsigned int)ldo))
          && NULL != (kernel = libxsmm_dispatch_trans(desc))) /* JIT-kernel available */
        {
          LIBXSMM_TCOPY_CALL(kernel, typesize, in, ldi, out, ldo);
        }
        else {
          LIBXSMM_XCOPY_NONJIT(LIBXSMM_TCOPY_KERNEL,
            typesize, out, in, ldi, ldo, 0, m, 0, n,
            LIBXSMM_XALIGN_TCOPY);
        }
      }
    }
    else if (ldi == ldo) {
      libxsmm_itrans/*TODO: omp*/(out, typesize, m, n, ldi);
    }
    else if (0 != libxsmm_get_verbosity() /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: output and input of the transpose must be different!\n");
    }
  }
#if defined(LIBXSMM_TRANS_CHECK)
  else {
    if (0 != libxsmm_get_verbosity() /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (0 == out || 0 == in) {
        fprintf(stderr, "LIBXSMM ERROR: the transpose input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXSMM ERROR: output and input of the transpose must be different!\n");
      }
      else if (0 == typesize) {
        fprintf(stderr, "LIBXSMM ERROR: the typesize of the transpose is zero!\n");
      }
      else if (0 >= m || 0 >= n) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix extent(s) of the transpose is/are zero or negative!\n");
      }
      else {
        LIBXSMM_ASSERT(ldi < m || ldo < n);
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the transpose is/are too small!\n");
      }
    }
  }
#endif
}


#if defined(LIBXSMM_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_otrans_omp)(void* /*out*/, const void* /*in*/, const unsigned int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_otrans_omp)(void* out, const void* in, const unsigned int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  LIBXSMM_ASSERT(0 != typesize && 0 != m);
  ldx = *(ldi ? ldi : m);
  libxsmm_otrans_omp(out, in, *typesize, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}

#endif /*defined(LIBXSMM_BUILD)*/

