/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#include <libxsmm.h>

/* external implementation, if a supported library is enabled at build-time */
#if !defined(LIBXSMM_TRANSPOSE_EXTERNAL) &&
   ((defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) \
  || defined(__OPENBLAS))
/*# define LIBXSMM_TRANSPOSE_EXTERNAL*/
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdio.h>
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(LIBXSMM_TRANSPOSE_EXTERNAL)
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
#   include <mkl_trans.h>
# elif defined(__OPENBLAS)
#   include <openblas/cblas.h>
# endif
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_TRANSPOSE_CACHESIZE)
# define LIBXSMM_TRANSPOSE_CACHESIZE 32768
#endif
#if !defined(LIBXSMM_TRANSPOSE_CHUNK)
# define LIBXSMM_TRANSPOSE_CHUNK 32
#endif
#if !defined(LIBXSMM_TRANSPOSE_CONSECUTIVE_STORE)
# define LIBXSMM_TRANSPOSE_CONSECUTIVE_STORE
#endif

/* consecutive store and strided load */
#if defined(LIBXSMM_TRANSPOSE_CONSECUTIVE_STORE)
# define INTERNAL_TRANSPOSE_INDEX_STORE(I, J, LD) (I * LD + J)
# define INTERNAL_TRANSPOSE_INDEX_LOAD(I, J, LD) (J * LD + I)
#else /* consecutive load and strided store */
# define INTERNAL_TRANSPOSE_INDEX_STORE(I, J, LD) (J * LD + I)
# define INTERNAL_TRANSPOSE_INDEX_LOAD(I, J, LD) (I * LD + J)
#endif

#define INTERNAL_TRANSPOSE_OOP(TYPE, OUT, IN, M0, M1, N0, N1, N, LD, LDO) { \
  const TYPE *const a = (const TYPE*)IN; \
  TYPE *const b = (TYPE*)OUT; \
  libxsmm_blasint i, j; \
  if (LIBXSMM_TRANSPOSE_CHUNK == N && 0 == LIBXSMM_MOD2((uintptr_t)b, LIBXSMM_ALIGNMENT)) { \
    for (i = M0; i < M1; ++i) { \
      LIBXSMM_PRAGMA_VALIGNED_VARS(b) \
      LIBXSMM_PRAGMA_NONTEMPORAL \
      for (j = N0; j < N0 + LIBXSMM_TRANSPOSE_CHUNK; ++j) { \
        b[INTERNAL_TRANSPOSE_INDEX_STORE(i,j,LDO)] = a[INTERNAL_TRANSPOSE_INDEX_LOAD(i,j,LD)]; \
      } \
    } \
  } \
  else { \
    for (i = M0; i < M1; ++i) { \
      LIBXSMM_PRAGMA_NONTEMPORAL \
      for (j = N0; j < N1; ++j) { \
        b[INTERNAL_TRANSPOSE_INDEX_STORE(i,j,LDO)] = a[INTERNAL_TRANSPOSE_INDEX_LOAD(i,j,LD)]; \
      } \
    } \
  } \
}


/* Based on cache-oblivious scheme as published by Frigo et.al. Further optimization for loop with bounds known at compile-time. */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_transpose_oop(void *LIBXSMM_RESTRICT out, const void *LIBXSMM_RESTRICT in,
  unsigned int typesize, libxsmm_blasint m0, libxsmm_blasint m1, libxsmm_blasint n0, libxsmm_blasint n1,
  libxsmm_blasint ld, libxsmm_blasint ldo)
{
  const libxsmm_blasint m = m1 - m0, n = n1 - n0;

  if (m * n * typesize <= (LIBXSMM_TRANSPOSE_CACHESIZE / 2)) {
    switch(typesize) {
      case 1: {
        INTERNAL_TRANSPOSE_OOP(char, out, in, m0, m1, n0, n1, n, ld, ldo);
      } break;
      case 2: {
        INTERNAL_TRANSPOSE_OOP(short, out, in, m0, m1, n0, n1, n, ld, ldo);
      } break;
      case 4: {
        INTERNAL_TRANSPOSE_OOP(float, out, in, m0, m1, n0, n1, n, ld, ldo);
      } break;
      case 8: {
        INTERNAL_TRANSPOSE_OOP(double, out, in, m0, m1, n0, n1, n, ld, ldo);
      } break;
      case 16: {
        typedef struct dvec2_t { double value[2]; } dvec2_t;
        INTERNAL_TRANSPOSE_OOP(dvec2_t, out, in, m0, m1, n0, n1, n, ld, ldo);
      } break;
      default: {
#if !defined(NDEBUG) /* library code is expected to be mute */
        fprintf(stderr, "LIBXSMM: unsupported element type in transpose!\n");
#endif
        assert(0);
      }
    }
  }
  else if (m >= n) {
    const libxsmm_blasint mi = (m0 + m1) / 2;
    internal_transpose_oop(out, in, typesize, m0, mi, n0, n1, ld, ldo);
    internal_transpose_oop(out, in, typesize, mi, m1, n0, n1, ld, ldo);
  }
  else {
#if (0 < LIBXSMM_TRANSPOSE_CHUNK)
    if (LIBXSMM_TRANSPOSE_CHUNK < n) {
      const libxsmm_blasint ni = n0 + LIBXSMM_TRANSPOSE_CHUNK;
      internal_transpose_oop(out, in, typesize, m0, m1, n0, ni, ld, ldo);
      internal_transpose_oop(out, in, typesize, m0, m1, ni, n1, ld, ldo);
    }
    else
#endif
    {
      const libxsmm_blasint ni = (n0 + n1) / 2;
      internal_transpose_oop(out, in, typesize, m0, m1, n0, ni, ld, ldo);
      internal_transpose_oop(out, in, typesize, m0, m1, ni, n1, ld, ldo);
    }
  }
}


LIBXSMM_API_DEFINITION void libxsmm_transpose_oop(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, libxsmm_blasint ldo)
{
#if !defined(NDEBUG) /* library code is expected to be mute */
  if (ld < m && ldo < n) {
    fprintf(stderr, "LIBXSMM: the leading dimensions of the transpose are too small!\n");
  }
  else if (ld < m) {
    fprintf(stderr, "LIBXSMM: the leading dimension of the transpose input is too small!\n");
  }
  else if (ldo < n) {
    fprintf(stderr, "LIBXSMM: the leading dimension of the transpose output is too small!\n");
  }
#endif
#if defined(LIBXSMM_TRANSPOSE_EXTERNAL)
  if (8 == typesize) { /* hopefully the actual type is not complex-SP (or alpha-multiplication is not performed) */
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
    mkl_domatcopy('C', 'T', m, n, 1.0, (const double*)in, ld, (double*)out, ldo);
# elif defined(__OPENBLAS) /* tranposes are not really covered by the common CBLAS interface */
    cblas_domatcopy(CblasColMajor, CblasTrans, m, n, 1.0, (const double*)in, ld, (double*)out, ldo);
# endif
  }
  else if (4 == typesize) {
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
    mkl_somatcopy('C', 'T', m, n, 1.f, (const float*)in, ld, (float*)out, ldo);
# elif defined(__OPENBLAS) /* tranposes are not really covered by the common CBLAS interface */
    cblas_somatcopy(CblasColMajor, CblasTrans, m, n, 1.f, (const float*)in, ld, (float*)out, ldo);
# endif
  }
  else if (16 == typesize) {
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
    const MKL_Complex16 one = { 1.0/*real*/, 0.0/*imag*/ };
    mkl_zomatcopy('C', 'T', m, n, one, (const MKL_Complex16*)in, ld, (MKL_Complex16*)out, ldo);
# elif defined(__OPENBLAS) /* tranposes are not really covered by the common CBLAS interface */
    cblas_zomatcopy(CblasColMajor, CblasTrans, m, n, 1.0, (const double*)in, ld, (double*)out, ldo);
# endif
  }
  else
#endif
  {
    internal_transpose_oop(out, in, typesize, 0, m, 0, n, ld, ldo);
  }
}


LIBXSMM_API_DEFINITION void libxsmm_transpose_inp(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  LIBXSMM_UNUSED(inout); LIBXSMM_UNUSED(typesize); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(ld);
  assert(0/*Not yet implemented!*/);
}


#if defined(LIBXSMM_BUILD)

LIBXSMM_API_DEFINITION void libxsmm_stranspose_oop(float* out, const float* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, libxsmm_blasint ldo)
{
  libxsmm_transpose_oop(out, in, sizeof(float), m, n, ld, ldo);
}


LIBXSMM_API_DEFINITION void libxsmm_dtranspose_oop(double* out, const double* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, libxsmm_blasint ldo)
{
  libxsmm_transpose_oop(out, in, sizeof(double), m, n, ld, ldo);
}


LIBXSMM_API_DEFINITION void libxsmm_stranspose_inp(float* inout,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  libxsmm_transpose_inp(inout, sizeof(float), m, n, ld);
}


LIBXSMM_API_DEFINITION void libxsmm_dtranspose_inp(double* inout,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  libxsmm_transpose_inp(inout, sizeof(double), m, n, ld);
}

#endif /*defined(LIBXSMM_BUILD)*/

