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
#include "libxsmm_main.h"
#include <libxsmm_cpuid.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_DEFINITION void libxsmm_trans_init(int archid)
{
  libxsmm_trans_chunksize = LIBXSMM_TRANS_MAX_CHUNKSIZE;
#if defined(__MIC__)
  LIBXSMM_UNUSED(archid);
#else
  if (LIBXSMM_X86_AVX512_MIC == archid)
#endif
  {
    libxsmm_trans_chunksize = LIBXSMM_TRANS_MIN_CHUNKSIZE;
  }
}


LIBXSMM_API_DEFINITION void libxsmm_trans_finalize(void)
{
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_otrans(void *LIBXSMM_RESTRICT out, const void *LIBXSMM_RESTRICT in,
  unsigned int typesize, libxsmm_blasint m0, libxsmm_blasint m1, libxsmm_blasint n0, libxsmm_blasint n1,
  libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  LIBXSMM_OTRANS_MAIN(LIBXSMM_NOOP_ARGS, internal_otrans, out, in, typesize, m0, m1, n0, n1, ldi, ldo);
}


LIBXSMM_API_DEFINITION int libxsmm_otrans(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  assert(0 < typesize);
  if (ldi >= m && ldo >= n && 0 != out && 0 != in) {
    LIBXSMM_INIT
    if (out != in) {
      internal_otrans(out, in, typesize, 0, m, 0, n, ldi, ldo);
    }
    else if (ldi == ldo) {
      result = libxsmm_itrans(out, typesize, m, n, ldi);
    }
    else {
      if (0 != libxsmm_verbosity /* library code is expected to be mute */
       && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM: output location of the transpose must be different from the input!\n");
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
    result = EXIT_FAILURE;
  }

  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_itrans(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (0 != inout) {
    LIBXSMM_INIT
    if (m == n) { /* some fallback; still warned as "not implemented" */
      libxsmm_blasint i, j;
      for (i = 0; i < n; ++i) {
        for (j = 0; j < i; ++j) {
          char *const a = ((char*)inout) + (i * ld + j) * typesize;
          char *const b = ((char*)inout) + (j * ld + i) * typesize;
          unsigned int k;
          for (k = 0; k < typesize; ++k) {
            const char tmp = a[k];
            a[k] = b[k];
            b[k] = tmp;
          }
        }
      }
    }
    else {
      if (0 != libxsmm_verbosity /* library code is expected to be mute */
       && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM: in-place transpose is not fully implemented!\n");
      }
      assert(0/*TODO: proper implementation is pending*/);
      result = EXIT_FAILURE;
    }
    if ((1 < libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM: performance warning - in-place transpose is not fully implemented!\n");
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM: the transpose input/output is NULL!\n");
    }
    result = EXIT_FAILURE;
  }

  return result;
}


#if defined(LIBXSMM_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_otrans)(void* /*out*/, const void* /*in*/, const unsigned int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_otrans)(void* out, const void* in, const unsigned int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  assert(0 != typesize && 0 != m);
  ldx = *(0 != ldi ? ldi : m);
  libxsmm_otrans(out, in, *typesize, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}


LIBXSMM_API_DEFINITION int libxsmm_sotrans(float* out, const float* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_otrans(out, in, sizeof(float), m, n, ldi, ldo);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sotrans)(float* /*out*/, const float* /*in*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_sotrans)(float* out, const float* in,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  assert(0 != m);
  ldx = *(0 != ldi ? ldi : m);
  libxsmm_sotrans(out, in, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}


LIBXSMM_API_DEFINITION int libxsmm_dotrans(double* out, const double* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_otrans(out, in, sizeof(double), m, n, ldi, ldo);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dotrans)(double* /*out*/, const double* /*in*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_dotrans)(double* out, const double* in,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  assert(0 != m);
  ldx = *(0 != ldi ? ldi : m);
  libxsmm_dotrans(out, in, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* /*inout*/, const unsigned int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ld*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* inout, const unsigned int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ld)
{
  assert(0 != typesize && 0 != m);
  libxsmm_itrans(inout, *typesize, *m, *(n ? n : m), *(0 != ld ? ld : m));
}


LIBXSMM_API_DEFINITION int libxsmm_sitrans(float* inout, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  return libxsmm_itrans(inout, sizeof(float), m, n, ld);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sitrans)(float* /*inout*/, const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ld*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_sitrans)(float* inout, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ld)
{
  assert(0 != m);
  libxsmm_sitrans(inout, *m, *(n ? n : m), *(0 != ld ? ld : m));
}


LIBXSMM_API_DEFINITION int libxsmm_ditrans(double* inout, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  return libxsmm_itrans(inout, sizeof(double), m, n, ld);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_ditrans)(double* /*inout*/, const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ld*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_ditrans)(double* inout, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ld)
{
  assert(0 != m);
  libxsmm_ditrans(inout, *m, *(n ? n : m), *(0 != ld ? ld : m));
}

#endif /*defined(LIBXSMM_BUILD)*/

