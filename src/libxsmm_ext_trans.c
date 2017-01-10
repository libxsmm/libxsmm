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
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define LIBXSMM_EXT_TRANS_MT_THRESHOLD (LIBXSMM_MAX_MNK / LIBXSMM_AVG_K)


#if defined(LIBXSMM_EXT_TASKS)
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_ext_otrans(void *LIBXSMM_RESTRICT out, const void *LIBXSMM_RESTRICT in,
  unsigned int typesize, libxsmm_blasint m0, libxsmm_blasint m1, libxsmm_blasint n0, libxsmm_blasint n1,
  libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  LIBXSMM_OTRANS_MAIN(LIBXSMM_EXT_TSK_KERNEL_VARS, internal_ext_otrans, out, in, typesize, m0, m1, n0, n1, ldi, ldo);
}
#endif /*defined(LIBXSMM_EXT_TASKS)*/


LIBXSMM_API_DEFINITION int libxsmm_otrans_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG) /* library code is expected to be mute */
  static int error_once = 0;
#endif
  assert(0 < typesize);
  if (ldi >= m && ldo >= n && 0 != out && 0 != in) {
    LIBXSMM_INIT
    if (out != in) {
#if defined(LIBXSMM_EXT_TASKS)
      if (0 != libxsmm_mt /* enable OpenMP support, ... */
        /* ... but consider a threshold of the problem-size */
        && ((LIBXSMM_EXT_TRANS_MT_THRESHOLD) < (m * n)))
      {
        if (0 == LIBXSMM_MOD2(libxsmm_mt, 2)) { /* even: enable internal parallelization */
          LIBXSMM_EXT_TSK_PARALLEL_ONLY
          internal_ext_otrans(out, in, typesize, 0, m, 0, n, ldi, ldo);
          /* implicit synchronization (barrier) */
        }
        else { /* odd: prepare for external parallelization */
          LIBXSMM_EXT_SINGLE
          internal_ext_otrans(out, in, typesize, 0, m, 0, n, ldi, ldo);
          if (1 == libxsmm_mt) { /* allow to omit synchronization */
            LIBXSMM_EXT_TSK_SYNC
          }
        }
      }
      else
#endif
      {
        libxsmm_otrans(out, in, typesize, m, n, ldi, ldo);
      }
    }
    else if (ldi == ldo) {
      result = libxsmm_itrans/*TODO: omp*/(out, typesize, m, n, ldi);
    }
    else {
#if !defined(NDEBUG) /* library code is expected to be mute */
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXSMM: output location of the transpose must be different from the input!\n");
      }
#endif
      result = EXIT_FAILURE;
    }
  }
  else {
#if !defined(NDEBUG) /* library code is expected to be mute */
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      if (0 == out || 0 == in) {
        fprintf(stderr, "LIBXSMM: the transpose input and/or output is NULL!\n");
      }
      else if (ldi < m && ldo < n) {
        fprintf(stderr, "LIBXSMM: the leading dimensions of the transpose are too small!\n");
      }
      else if (ldi < m) {
        fprintf(stderr, "LIBXSMM: the leading dimension of the transpose input is too small!\n");
      }
      else {
        assert(ldo < n);
        fprintf(stderr, "LIBXSMM: the leading dimension of the transpose output is too small!\n");
      }
    }
#endif
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

