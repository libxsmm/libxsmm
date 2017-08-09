/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_DEFINITION int libxsmm_matcopy_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  assert(typesize <= 255);
  if (0 != out && out != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && m <= ldo) {
    const unsigned int size = (unsigned int)(1U * m * n);
    if ((LIBXSMM_TRANS_THRESHOLD) < size) { /* consider problem-size (threshold) */
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
      if (0 == omp_get_active_level())
#endif
      { /* enable internal parallelization */
        LIBXSMM_EXT_PARALLEL
        {
#if defined(_OPENMP)
          const int tid = omp_get_thread_num(), nthreads = omp_get_num_threads();
#else
          const int tid = 0, nthreads = 1;
#endif
          libxsmm_matcopy_thread(out, in, typesize, m, n, ldi, ldo, prefetch, tid, nthreads);
        } /* implicit synchronization (barrier) */
      }
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
      else { /* assume external parallelization */
        const int tindex = (4 < typesize ? 0 : 1), index = LIBXSMM_MIN(LIBXSMM_SQRT2(size) >> 10, 7);
        const unsigned int uldi = (unsigned int)ldi, uldo = (unsigned int)ldo;
        libxsmm_matcopy_descriptor descriptor = { 0 };
        libxsmm_xmatcopyfunction xmatcopy = 0;
        LIBXSMM_INIT
        descriptor.m = LIBXSMM_MIN(libxsmm_trans_tile[tindex][0/*M*/][index], (unsigned int)m);
        descriptor.n = LIBXSMM_MIN(libxsmm_trans_tile[tindex][1/*N*/][index], (unsigned int)n);
        descriptor.prefetch = (unsigned char)((0 == prefetch || 0 == *prefetch) ? 0 : 1);
        if (0 != (1 & libxsmm_trans_jit)) { /* JIT'ted matcopy permitted? */
          descriptor.flags = (unsigned char)(0 != in ? 0 : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE);
          descriptor.typesize = (unsigned char)typesize; descriptor.unroll_level = 2;
          descriptor.ldi = (unsigned int)ldi; descriptor.ldo = (unsigned int)ldo;
          xmatcopy = libxsmm_xmatcopydispatch(&descriptor);
        }
        if (0 == xmatcopy || 0 == descriptor.prefetch) {
          LIBXSMM_XCOPY(
            LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_EXT_TSK_KERNEL_ARGS,
            if (0 != libxsmm_sync) { LIBXSMM_EXT_TSK_SYNC } /* allow to omit synchronization */,
            LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL_NOPF, xmatcopy,
            out, in, typesize, uldi, uldo, descriptor.m, descriptor.n, 0, m, 0, n);
        }
        else {
          LIBXSMM_XCOPY(
            LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_EXT_TSK_KERNEL_ARGS,
            if (0 != libxsmm_sync) { LIBXSMM_EXT_TSK_SYNC } /* allow to omit synchronization */,
            LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL, xmatcopy,
            out, in, typesize, uldi, uldo, descriptor.m, descriptor.n, 0, m, 0, n);
        }
      }
#endif
    }
    else { /* small problem-size (no MT) */
      result = libxsmm_matcopy(out, in, typesize, m, n, ldi, ldo, prefetch);
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
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
        assert(ldi < m || ldo < n);
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the matcopy is/are too small!\n");
      }
    }
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_otrans_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  assert(typesize <= 255);
  if (0 != out && 0 != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && n <= ldo) {
    if (out != in) {
      const unsigned int size = (unsigned int)(1U * m * n);
      if ((LIBXSMM_TRANS_THRESHOLD) < size) { /* consider problem-size (threshold) */
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
        if (0 == omp_get_active_level())
#endif
        { /* enable internal parallelization */
          LIBXSMM_EXT_PARALLEL
          {
#if defined(_OPENMP)
            const int tid = omp_get_thread_num(), nthreads = omp_get_num_threads();
#else
            const int tid = 0, nthreads = 1;
#endif
            libxsmm_otrans_thread(out, in, typesize, m, n, ldi, ldo, tid, nthreads);
          } /* implicit synchronization (barrier) */
        }
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
        else { /* assume external parallelization */
          const int tindex = (4 < typesize ? 0 : 1), index = LIBXSMM_MIN(LIBXSMM_SQRT2(size) >> 10, 7);
          const unsigned int uldi = (unsigned int)ldi, uldo = (unsigned int)ldo;
          libxsmm_transpose_descriptor descriptor = { 0 };
          libxsmm_xtransfunction xtrans = 0;
          LIBXSMM_INIT
          descriptor.m = LIBXSMM_MIN(libxsmm_trans_tile[tindex][0/*M*/][index], (unsigned int)m);
          descriptor.n = LIBXSMM_MIN(libxsmm_trans_tile[tindex][1/*N*/][index], (unsigned int)n);
          if (0 != (2 & libxsmm_trans_jit)) { /* JIT'ted transpose */
            descriptor.typesize = (unsigned char)typesize; descriptor.ldo = (unsigned int)ldo;
            descriptor.m = LIBXSMM_MIN(descriptor.m, LIBXSMM_MAX_M);
            descriptor.n = LIBXSMM_MIN(descriptor.n, LIBXSMM_MAX_N);
            xtrans = libxsmm_xtransdispatch(&descriptor);
          }
          LIBXSMM_XCOPY(
            LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_EXT_TSK_KERNEL_ARGS,
            if (0 != libxsmm_sync) { LIBXSMM_EXT_TSK_SYNC } /* allow to omit synchronization */,
            LIBXSMM_TCOPY_KERNEL, LIBXSMM_TCOPY_CALL, xtrans,
            out, in, typesize, uldi, uldo, descriptor.m, descriptor.n, 0, m, 0, n);
        }
#endif
      }
      else { /* small problem-size (no MT) */
        result = libxsmm_otrans(out, in, typesize, m, n, ldi, ldo);
      }
    }
    else if (ldi == ldo) {
      result = libxsmm_itrans/*TODO: omp*/(out, typesize, m, n, ldi);
    }
    else {
      if (0 != libxsmm_verbosity /* library code is expected to be mute */
       && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: output and input of the transpose must be different!\n");
      }
      result = EXIT_FAILURE;
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
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
        assert(ldi < m || ldo < n);
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the transpose is/are too small!\n");
      }
    }
    result = EXIT_FAILURE;
  }

  return result;
}


#if defined(LIBXSMM_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_otrans_omp)(void* /*out*/, const void* /*in*/, const unsigned int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_otrans_omp)(void* out, const void* in, const unsigned int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  assert(0 != typesize && 0 != m);
  ldx = *(ldi ? ldi : m);
  libxsmm_otrans_omp(out, in, *typesize, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}

#endif /*defined(LIBXSMM_BUILD)*/

