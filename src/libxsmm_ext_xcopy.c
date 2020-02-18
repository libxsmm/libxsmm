/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_xcopy.h"
#include "libxsmm_ext.h"

#define LIBXSMM_MCOPY_MT(MT, NT, M, N) ((MT) <= (M) && (NT) <= (N) && (64U * 64U) <= (((unsigned int)(M)) * (N)))


LIBXSMM_APIEXT void libxsmm_matcopy_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch)
{
  if (0 < typesize && m <= ldi && m <= ldo && out != in &&
    ((NULL != out && 0 < m && 0 < n) || (0 == m && 0 == n)))
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
            NULL != in ? 0 : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE, iprefetch, NULL/*default unroll*/)))
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
# if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
          const int nthreads = omp_get_num_threads();
          const int ntasks = (0 == libxsmm_trans_taskscale
            ? (LIBXSMM_XCOPY_TASKSCALE)
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
            prefetch, tm, tn, kernel, 0/*tid*/, 1/*nthreads*/);
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
  else {
    static int error_once = 0;
    if ( 0 != libxsmm_get_verbosity() /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (0 == out) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix-copy input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXSMM ERROR: output and input of the matrix-copy must be different!\n");
      }
      else if (0 == typesize) {
        fprintf(stderr, "LIBXSMM ERROR: the type-size of the matrix-copy is zero!\n");
      }
      else if (ldi < m || ldo < m) {
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the matrix-copy is/are too small!\n");
      }
      else if (0 > m || 0 > n) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix extent(s) of the matrix-copy is/are negative!\n");
      }
    }
  }
}


LIBXSMM_APIEXT void libxsmm_otrans_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  static int error_once = 0;
  if (0 < typesize && m <= ldi && n <= ldo &&
    ((NULL != out && NULL != in && 0 < m && 0 < n) || (0 == m && 0 == n)))
  {
    LIBXSMM_INIT
    if (out != in) {
#if defined(_OPENMP)
      const unsigned int tm = libxsmm_trans_mtile[4 < typesize ? 0 : 1];
      const unsigned int tn = (unsigned int)(libxsmm_trans_tile_stretch * tm);
      if (tm <= (unsigned int)m && tn <= (unsigned int)n) { /* consider problem-size */
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
            { /* coverity[divide_by_zero] */
              libxsmm_otrans_thread_internal(out, in, typesize,
                (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
                tm, tn, NULL/*kernel*/, omp_get_thread_num(), nthreads);
            }
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
# if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
          const int nthreads = omp_get_num_threads();
          const int ntasks = (0 == libxsmm_trans_taskscale
            ? (LIBXSMM_XCOPY_TASKSCALE)
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
# else    /* coverity[divide_by_zero] */
          libxsmm_otrans_thread_internal(out, in, typesize,
            (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
            tm, tn, NULL/*kernel*/, 0/*tid*/, 1/*nthreads*/);
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
        fprintf(stderr, "LIBXSMM ERROR: the type-size of the transpose is zero!\n");
      }
      else if (ldi < m || ldo < n) {
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the transpose is/are too small!\n");
      }
      else if (0 > m || 0 > n) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix extent(s) of the transpose is/are negative!\n");
      }
    }
  }
}


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_otrans_omp)(void* /*out*/, const void* /*in*/, const int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_otrans_omp)(void* out, const void* in, const int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  LIBXSMM_ASSERT(NULL != typesize && 0 < *typesize && NULL != m);
  ldx = *(NULL != ldi ? ldi : m);
  libxsmm_otrans_omp(out, in, (unsigned int)*typesize, *m, *(NULL != n ? n : m), ldx, NULL != ldo ? *ldo : ldx);
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/

