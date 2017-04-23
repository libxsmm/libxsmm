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

#define LIBXSMM_EXT_TRANS_MT_THRESHOLD (LIBXSMM_MAX_MNK / LIBXSMM_AVG_K)


#if defined(LIBXSMM_EXT_TASKS)

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_matcopy_nopf_omp(libxsmm_xmatcopyfunction xmatcopy,
  void *LIBXSMM_RESTRICT out, const void *LIBXSMM_RESTRICT in, unsigned int typesize,
  unsigned int ldi, unsigned int ldo, unsigned int tile_m, unsigned int tile_n,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1)
{
  LIBXSMM_XCOPY_RECURSIVE(internal_matcopy_nopf_omp,
    LIBXSMM_EXT_TSK_KERNEL_ARGS, LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL, xmatcopy,
    out, in, typesize, ldi, ldo, tile_m, tile_n, m0, m1, n0, n1);
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_matcopy_omp(libxsmm_xmatcopyfunction xmatcopy,
  void *LIBXSMM_RESTRICT out, const void *LIBXSMM_RESTRICT in, unsigned int typesize,
  unsigned int ldi, unsigned int ldo, unsigned int tile_m, unsigned int tile_n,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1)
{
  LIBXSMM_XCOPY_RECURSIVE(internal_matcopy_omp,
    LIBXSMM_EXT_TSK_KERNEL_ARGS, LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL, xmatcopy,
    out, in, typesize, ldi, ldo, tile_m, tile_n, m0, m1, n0, n1);
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_otrans_omp(libxsmm_xtransfunction xtrans,
  void *LIBXSMM_RESTRICT out, const void *LIBXSMM_RESTRICT in, unsigned int typesize,
  unsigned int ldi, unsigned int ldo, unsigned int tile_m, unsigned int tile_n,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1)
{
  LIBXSMM_XCOPY_RECURSIVE(internal_otrans_omp,
    LIBXSMM_EXT_TSK_KERNEL_ARGS, LIBXSMM_TCOPY_KERNEL, LIBXSMM_TCOPY_CALL, xtrans,
    out, in, typesize, ldi, ldo, tile_m, tile_n, m0, m1, n0, n1);
}

#endif /*defined(LIBXSMM_EXT_TASKS)*/


LIBXSMM_API_DEFINITION int libxsmm_matcopy_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  assert(typesize <= 255);
  if (0 != out && out != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && n <= ldo) {
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
    if ((LIBXSMM_EXT_TRANS_MT_THRESHOLD) < (m * n)) { /* consider problem-size (threshold) */
      const int tindex = (4 < typesize ? 0 : 1), index = LIBXSMM_MIN(LIBXSMM_SQRT2(1ULL * m * n) >> 10, 7);
      libxsmm_matcopy_descriptor descriptor = { 0 };
      libxsmm_xmatcopyfunction xmatcopy = 0;
      LIBXSMM_INIT
      descriptor.m = LIBXSMM_MIN((unsigned int)m, libxsmm_trans_tile[tindex][0/*M*/][index]);
      descriptor.n = LIBXSMM_MIN((unsigned int)n, libxsmm_trans_tile[tindex][1/*N*/][index]);
      descriptor.prefetch = (unsigned char)((0 == prefetch || 0 == *prefetch) ? 0 : 1);
      if (0 != (1 & libxsmm_trans_jit)) { /* JIT'ted matcopy */
        descriptor.ldi = ldi; descriptor.ldo = ldo; descriptor.unroll_level = 1;
        descriptor.typesize = (unsigned char)typesize;
        descriptor.flags = (unsigned char)(0 != in ? 0 : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE);
        xmatcopy = libxsmm_xmatcopydispatch(&descriptor);
      }
      if (0 == xmatcopy || 0 == descriptor.prefetch) {
        internal_matcopy_nopf_omp(xmatcopy, out, in, typesize, ldi, ldo, descriptor.m, descriptor.n, 0, m, 0, n);
      }
      else {
        internal_matcopy_omp(xmatcopy, out, in, typesize, ldi, ldo, descriptor.m, descriptor.n, 0, m, 0, n);
      }
    }
    else
#endif /*defined(LIBXSMM_EXT_TASKS)*/
    {
      libxsmm_matcopy(out, in, typesize, m, n, ldi, ldo, prefetch);
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (0 == out) {
        fprintf(stderr, "LIBXSMM: the matcopy input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXSMM: output and input of the matcopy must be different!\n");
      }
      else if (0 == typesize) {
        fprintf(stderr, "LIBXSMM: the typesize of the matcopy is zero!\n");
      }
      else if (0 >= m || 0 >= n) {
        fprintf(stderr, "LIBXSMM: the matrix extent(s) of the matcopy is/are zero or negative!\n");
      }
      else {
        assert(ldi < m || ldo < n);
        fprintf(stderr, "LIBXSMM: the leading dimension(s) of the matcopy is/are too small!\n");
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

  if (0 != out && 0 != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && n <= ldo) {
    LIBXSMM_INIT
    if (out != in) {
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
      if ((LIBXSMM_EXT_TRANS_MT_THRESHOLD) < (m * n)) { /* consider problem-size (threshold) */
        libxsmm_xtransfunction xtrans = 0;
        libxsmm_transpose_descriptor descriptor = { 0 };
        const int tindex = (4 < typesize ? 0 : 1), index = LIBXSMM_MIN(LIBXSMM_SQRT2(1ULL * m * n) >> 10, 7);
        descriptor.m = LIBXSMM_MIN((unsigned int)m, libxsmm_trans_tile[tindex][0/*M*/][index]);
        descriptor.n = LIBXSMM_MIN((unsigned int)n, libxsmm_trans_tile[tindex][1/*N*/][index]);
        if (0 != (2 & libxsmm_trans_jit)) { /* JIT'ted transpose */
          descriptor.typesize = typesize; descriptor.ldo = ldo;
          xtrans = libxsmm_xtransdispatch(&descriptor);
        }
        if (0 == omp_get_level()) { /* enable internal parallelization */
          LIBXSMM_EXT_TSK_PARALLEL
          internal_otrans_omp(xtrans, out, in, typesize, ldi, ldo, descriptor.m, descriptor.n, 0, m, 0, n);
          /* implicit synchronization (barrier) */
        }
        else { /* assume external parallelization */
          internal_otrans_omp(xtrans, out, in, typesize, ldi, ldo, descriptor.m, descriptor.n, 0, m, 0, n);
          /* allow to omit synchronization */
          if (0 != libxsmm_sync) {
            LIBXSMM_EXT_TSK_SYNC
          }
        }
      }
      else
#endif /*defined(LIBXSMM_EXT_TASKS)*/
      {
        libxsmm_otrans(out, in, typesize, m, n, ldi, ldo);
      }
    }
    else if (ldi == ldo) {
      result = libxsmm_itrans/*TODO: omp*/(out, typesize, m, n, ldi);
    }
    else {
      if (0 != libxsmm_verbosity /* library code is expected to be mute */
       && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM: output and input of the transpose must be different!\n");
      }
      result = EXIT_FAILURE;
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (0 == out || 0 == in) {
        fprintf(stderr, "LIBXSMM: the transpose input and/or output is NULL!\n");
      }
      else if (out == in) {
        fprintf(stderr, "LIBXSMM: output and input of the transpose must be different!\n");
      }
      else if (0 == typesize) {
        fprintf(stderr, "LIBXSMM: the typesize of the transpose is zero!\n");
      }
      else if (0 >= m || 0 >= n) {
        fprintf(stderr, "LIBXSMM: the matrix extent(s) of the transpose is/are zero or negative!\n");
      }
      else {
        assert(ldi < m || ldo < n);
        fprintf(stderr, "LIBXSMM: the leading dimension(s) of the transpose is/are too small!\n");
      }
    }
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_sotrans_omp(float* out, const float* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_otrans_omp(out, in, sizeof(float), m, n, ldi, ldo);
}


LIBXSMM_API_DEFINITION int libxsmm_dotrans_omp(double* out, const double* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_otrans_omp(out, in, sizeof(double), m, n, ldi, ldo);
}


#if defined(LIBXSMM_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_otrans_omp)(void*, const void*, const unsigned int*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_otrans_omp)(void* out, const void* in, const unsigned int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  assert(0 != typesize && 0 != m);
  ldx = *(ldi ? ldi : m);
  libxsmm_otrans_omp(out, in, *typesize, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sotrans_omp)(float*, const float*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_sotrans_omp)(float* out, const float* in,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  assert(0 != m);
  ldx = *(ldi ? ldi : m);
  libxsmm_sotrans_omp(out, in, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dotrans_omp)(double*, const double*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_dotrans_omp)(double* out, const double* in,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  assert(0 != m);
  ldx = *(ldi ? ldi : m);
  libxsmm_dotrans_omp(out, in, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}

#endif /*defined(LIBXSMM_BUILD)*/

