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
  /* setup tile sizes according to CPUID or environment (LIBXSMM_TRANS_M, LIBXSMM_TRANS_N) */
  const unsigned int tile_configs[/*configs*/][2/*DP/SP*/][2/*TILE_M,TILE_N*/][8/*size-range*/] = {
    /* generic (hsw) */
    { { { 5, 5, 72, 135, 115, 117, 113, 123 }, { 63,  80,  20,  12, 16, 14, 16, 32 } },   /* DP */
      { { 7, 6,  7,  32, 152, 138, 145, 109 }, { 73, 147, 141, 157, 32, 32, 32, 44 } } }, /* SP */
    /* mic (knl/knm) */
    { { { 5, 8,  8,   9,  20,  17,  20,  25 }, { 25,  60,  78,  58, 52, 56, 36, 31 } },   /* DP */
      { { 5, 8,  8,   9,  20,  17,  20,  25 }, { 25,  60,  78,  58, 52, 56, 36, 31 } } }, /* SP */
    /* core (skx) */
    { { { 5, 8,  8,   9,  20,  17,  20,  25 }, { 25,  60,  78,  58, 52, 56, 36, 31 } },   /* DP */
      { { 5, 8,  8,   9,  20,  17,  20,  25 }, { 25,  60,  78,  58, 52, 56, 36, 31 } } }  /* SP */
  };
  const char *const env_jit = getenv("LIBXSMM_TRANS_JIT");
  const char *const env_m = getenv("LIBXSMM_TRANS_M"), *const env_n = getenv("LIBXSMM_TRANS_N");
  const int trans_m = ((0 == env_m || 0 == *env_m) ? -1 : atoi(env_m));
  const int trans_n = ((0 == env_n || 0 == *env_n) ? -1 : atoi(env_n));
  int config, i;

  if (LIBXSMM_X86_AVX512_CORE <= archid) {
    config = 2;
  }
  else if (LIBXSMM_X86_AVX512_MIC <= archid && LIBXSMM_X86_AVX512_CORE > archid) {
    config = 1;
  }
  else {
    config = 0;
  }

  /* determine if JIT-kernels are used (0: none, 1: matcopy, 2: transpose, 3: matcopy+transpose). */
  libxsmm_trans_jit = ((0 == env_jit || 0 == *env_jit) ? 3 : atoi(env_jit));

  for (i = 0; i < 8; ++i) {
    /* environment-defined tile sizes apply for DP and SP */
    libxsmm_trans_tile[0/*DP*/][0/*M*/][i] = libxsmm_trans_tile[1/*SP*/][0/*M*/][i] = (unsigned int)LIBXSMM_MAX(trans_m, 0);
    libxsmm_trans_tile[0/*DP*/][1/*N*/][i] = libxsmm_trans_tile[1/*SP*/][1/*N*/][i] = (unsigned int)LIBXSMM_MAX(trans_n, 0);
    /* load predefined configuration if tile size is not setup by the environment */
    if (0 >= libxsmm_trans_tile[0/*DP*/][0/*M*/][i]) libxsmm_trans_tile[0][0][i] = tile_configs[config][0][0][i];
    if (0 >= libxsmm_trans_tile[0/*DP*/][1/*N*/][i]) libxsmm_trans_tile[0][1][i] = tile_configs[config][0][1][i];
    if (0 >= libxsmm_trans_tile[1/*SP*/][0/*M*/][i]) libxsmm_trans_tile[1][0][i] = tile_configs[config][1][0][i];
    if (0 >= libxsmm_trans_tile[1/*SP*/][1/*N*/][i]) libxsmm_trans_tile[1][1][i] = tile_configs[config][1][1][i];
  }
}


LIBXSMM_API_DEFINITION void libxsmm_trans_finalize(void)
{
}


LIBXSMM_API_DEFINITION int libxsmm_matcopy(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  assert(typesize <= 255);
  if (0 != out && out != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && n <= ldo) {
    const unsigned int uldi = ldi, uldo = ldo;
    libxsmm_xmatcopyfunction xmatcopy = 0;
    LIBXSMM_INIT
    if (0 != (1 & libxsmm_trans_jit)) { /* JIT'ted matcopy permitted; use no tiling */
      libxsmm_matcopy_descriptor descriptor = { 0 };
      descriptor.prefetch = (unsigned char)((0 == prefetch || 0 == *prefetch) ? 0 : 1);
      descriptor.flags = (unsigned char)(0 != in ? 0 : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE);
      descriptor.ldi = ldi; descriptor.ldo = ldo; descriptor.unroll_level = 2;
      descriptor.typesize = (unsigned char)typesize;
      descriptor.m = m; descriptor.n = n;
      xmatcopy = libxsmm_xmatcopydispatch(&descriptor);
    }
    if (0 != xmatcopy) {
      if (0 == prefetch || 0 == *prefetch) {
        LIBXSMM_MCOPY_CALL_NOPF(xmatcopy, typesize, in, &uldi, out, &uldo);
      }
      else {
        LIBXSMM_MCOPY_CALL(xmatcopy, typesize, in, &uldi, out, &uldo);
      }
    }
    else { /* no JIT; tiled matrix-copy */
      const int tindex = (4 < typesize ? 0 : 1), index = LIBXSMM_MIN(LIBXSMM_SQRT2(1U * m * n) >> 10, 7);
      const libxsmm_blasint tm = LIBXSMM_MIN((libxsmm_blasint)libxsmm_trans_tile[tindex][0/*M*/][index], m);
      const libxsmm_blasint tn = LIBXSMM_MIN((libxsmm_blasint)libxsmm_trans_tile[tindex][1/*N*/][index], n);
      LIBXSMM_XCOPY(
        LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
        LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL_NOPF, xmatcopy,
        out, in, typesize, uldi, uldo, tm, tn, 0, m, 0, n);
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


LIBXSMM_API_DEFINITION int libxsmm_otrans(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  assert(typesize <= 255);
  if (0 != out && 0 != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && n <= ldo) {
    LIBXSMM_INIT
    if (out != in) {
      libxsmm_xtransfunction xtrans = 0;
      libxsmm_transpose_descriptor descriptor = { 0 };
      const unsigned int uldi = ldi, uldo = ldo;
      const unsigned int size = 1U * m * n;
      if (size <= (LIBXSMM_TRANS_THRESHOLD)) { /* no tiling */
        if (0 != (2 & libxsmm_trans_jit)) { /* JIT'ted transpose permitted? */
          descriptor.typesize = (unsigned char)typesize;
          descriptor.m = m; descriptor.n = n; descriptor.ldo = ldo;
          xtrans = libxsmm_xtransdispatch(&descriptor);
        }
        if (0 != xtrans) { /* prefer JIT for small problems */
          LIBXSMM_TCOPY_CALL(xtrans, typesize, in, &uldi, out, &uldo);
        }
        else { /* JIT not available */
          LIBXSMM_XCOPY_NONJIT(LIBXSMM_TCOPY_KERNEL, out, in, typesize, uldi, uldo, 0, m, 0, n);
        }
      }
      else { /* tiled transpose */
        const int tindex = (4 < typesize ? 0 : 1), index = LIBXSMM_MIN(LIBXSMM_SQRT2(size) >> 10, 7);
        descriptor.m = LIBXSMM_MIN((unsigned int)m, libxsmm_trans_tile[tindex][0/*M*/][index]);
        descriptor.n = LIBXSMM_MIN((unsigned int)n, libxsmm_trans_tile[tindex][1/*N*/][index]);
        if (0 != (2 & libxsmm_trans_jit)) { /* JIT'ted transpose permitted? */
          descriptor.typesize = (unsigned char)typesize; descriptor.ldo = ldo;
          descriptor.m = LIBXSMM_MIN(descriptor.m, LIBXSMM_MAX_M);
          descriptor.n = LIBXSMM_MIN(descriptor.n, LIBXSMM_MAX_N);
          xtrans = libxsmm_xtransdispatch(&descriptor);
        }
        LIBXSMM_XCOPY(
          LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
          LIBXSMM_TCOPY_KERNEL, LIBXSMM_TCOPY_CALL, xtrans,
          out, in, typesize, uldi, uldo, descriptor.m, descriptor.n, 0, m, 0, n);
      }
    }
    else if (ldi == ldo) {
      result = libxsmm_itrans(out, typesize, m, n, ldi);
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


LIBXSMM_API_DEFINITION int libxsmm_sotrans(float* out, const float* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_otrans(out, in, sizeof(float), m, n, ldi, ldo);
}


LIBXSMM_API_DEFINITION int libxsmm_dotrans(double* out, const double* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_otrans(out, in, sizeof(double), m, n, ldi, ldo);
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


LIBXSMM_API_DEFINITION int libxsmm_sitrans(float* inout,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  return libxsmm_itrans(inout, sizeof(float), m, n, ld);
}


LIBXSMM_API_DEFINITION int libxsmm_ditrans(double* inout,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  return libxsmm_itrans(inout, sizeof(double), m, n, ld);
}


#if defined(LIBXSMM_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matcopy)(void* /*out*/, const void* /*in*/, const unsigned int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/,
  const int* /*prefetch*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_matcopy)(void* out, const void* in, const unsigned int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo,
  const int* prefetch)
{
  libxsmm_blasint ldx;
  assert(0 != typesize && 0 != m);
  ldx = *(0 != ldi ? ldi : m);
  libxsmm_matcopy(out, in, *typesize, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx, prefetch);
}


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


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sitrans)(float* /*inout*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ld*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_sitrans)(float* inout,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ld)
{
  assert(0 != m);
  libxsmm_sitrans(inout, *m, *(n ? n : m), *(0 != ld ? ld : m));
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_ditrans)(double* /*inout*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ld*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_ditrans)(double* inout,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ld)
{
  assert(0 != m);
  libxsmm_ditrans(inout, *m, *(n ? n : m), *(0 != ld ? ld : m));
}

#endif /*defined(LIBXSMM_BUILD)*/

