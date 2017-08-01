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
    { { { 5, 5, 72, 135, 115, 117, 113, 123 }, { 63,  80,  20,  12, 16, 14, 16,  32 } },   /* DP */
      { { 7, 6,  7,  32, 152, 138, 145, 109 }, { 73, 147, 141, 157, 32, 32, 32,  44 } } }, /* SP */
    /* mic (knl/knm) */
    { { { 5, 8,  8,   9,  20,  17,  20,  25 }, { 25,  60,  78,  58, 52, 56, 36,  31 } },   /* DP */
      { { 5, 8,  8,   9,  20,  17,  20,  25 }, { 25,  60,  78,  58, 52, 56, 36,  31 } } }, /* SP */
    /* core (skx) */
    { { { 2, 2,  1,   2,  46,  17,  20,  76 }, { 20,  16,  42,   9,  3, 56, 36, 158 } },   /* DP */
      { { 5, 8,  8,   9,  20,  17,  20,  25 }, { 25,  60,  78,  58, 52, 56, 36,  31 } } }  /* SP */
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
#if !defined(__clang__) || defined(__INTEL_COMPILER) /* TODO: investigate Clang specific issue */
  /* determine if JIT-kernels are used (0: none, 1: matcopy, 2: transpose, 3: matcopy+transpose). */
  libxsmm_trans_jit = ((0 == env_jit || 0 == *env_jit) ? 3 : atoi(env_jit));
#else
  LIBXSMM_UNUSED(env_jit);
#endif
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


LIBXSMM_API_DEFINITION int libxsmm_matcopy_thread(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch, int tid, int nthreads)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  assert(typesize <= 255);
  if (0 != out && out != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && m <= ldo &&
    /* use (signed) integer types, but check sanity of input */
    0 <= tid && tid < nthreads)
  {
    const unsigned int uldi = (unsigned int)ldi, uldo = (unsigned int)ldo;
    libxsmm_xmatcopyfunction xmatcopy = 0;
    LIBXSMM_INIT
    if (1 < nthreads) {
      libxsmm_blasint m0 = 0, n0 = 0, m1 = m, n1 = n;
      libxsmm_matcopy_descriptor descriptor = { 0 };
      const int tindex = (4 < typesize ? 0 : 1), index = LIBXSMM_MIN(LIBXSMM_SQRT2(1U * m * n) >> 10, 7);
      int mtasks;
      descriptor.m = LIBXSMM_MIN(libxsmm_trans_tile[tindex][0/*M*/][index], (unsigned int)m);
      descriptor.n = LIBXSMM_MIN(libxsmm_trans_tile[tindex][1/*N*/][index], (unsigned int)n);
      if (0 != (1 & libxsmm_trans_jit)) { /* JIT'ted matcopy permitted */
        descriptor.prefetch = (unsigned char)((0 == prefetch || 0 == *prefetch) ? 0 : 1);
        descriptor.flags = (unsigned char)(0 != in ? 0 : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE);
        descriptor.typesize = (unsigned char)typesize; descriptor.unroll_level = 2;
        descriptor.ldi = (unsigned int)ldi; descriptor.ldo = (unsigned int)ldo;
        xmatcopy = libxsmm_xmatcopydispatch(&descriptor);
      }
      mtasks = ((1 < nthreads) ? ((int)((m + descriptor.m - 1) / descriptor.m)) : 1);
      if (1 < mtasks && nthreads <= mtasks) { /* only parallelized over M */
        const int mc = (mtasks + nthreads - 1) / nthreads * descriptor.m;
        m0 = tid * mc; m1 = LIBXSMM_MIN(m0 + mc, m);
      }
      else if (1 < nthreads) {
        const int mc = descriptor.m, ntasks = (nthreads / mtasks);
        const int nc = (((n + ntasks - 1) / ntasks + descriptor.n - 1) / descriptor.n) * descriptor.n;
        const int mtid = tid / ntasks, ntid = tid - mtid * ntasks;
        m0 = mtid * mc; m1 = LIBXSMM_MIN(m0 + mc, m);
        n0 = ntid * nc; n1 = LIBXSMM_MIN(n0 + nc, n);
      }
      assert(((tid + 1) != nthreads) || (m1 == m && n1 == n));
      if (0 != prefetch && 0 != *prefetch) { /* prefetch */
        LIBXSMM_XCOPY(
          LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
          LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL, xmatcopy, out, in,
          typesize, uldi, uldo, descriptor.m, descriptor.n, m0, m1, n0, n1);
      }
      else { /* no prefetch */
        LIBXSMM_XCOPY(
          LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
          LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL_NOPF, xmatcopy, out, in,
          typesize, uldi, uldo, descriptor.m, descriptor.n, m0, m1, n0, n1);
      }
    }
    else {
      assert(0 == tid && 1 == nthreads);
      if (0 != (1 & libxsmm_trans_jit)) { /* JIT'ted matcopy permitted */
        libxsmm_matcopy_descriptor descriptor = { 0 };
        descriptor.prefetch = (unsigned char)((0 == prefetch || 0 == *prefetch) ? 0 : 1);
        descriptor.flags = (unsigned char)(0 != in ? 0 : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE);
        descriptor.ldi = (unsigned int)ldi; descriptor.ldo = (unsigned int)ldo; descriptor.unroll_level = 2;
        descriptor.typesize = (unsigned char)typesize;
        descriptor.m = (unsigned int)m; descriptor.n = (unsigned int)n;
        xmatcopy = libxsmm_xmatcopydispatch(&descriptor);
      }
      if (0 != xmatcopy) { /* JIT-kernel available */
        if (0 != prefetch && 0 != *prefetch) { /* prefetch */
          LIBXSMM_MCOPY_CALL(xmatcopy, typesize, in, &uldi, out, &uldo);
        }
        else { /* no prefetch */
          LIBXSMM_MCOPY_CALL_NOPF(xmatcopy, typesize, in, &uldi, out, &uldo);
        }
      }
      else { /* no JIT */
        const int tindex = (4 < typesize ? 0 : 1), index = LIBXSMM_MIN(LIBXSMM_SQRT2(1U * m * n) >> 10, 7);
        const unsigned int tm = LIBXSMM_MIN(libxsmm_trans_tile[tindex][0/*M*/][index], (unsigned int)m);
        const unsigned int tn = LIBXSMM_MIN(libxsmm_trans_tile[tindex][1/*N*/][index], (unsigned int)n);
        assert(0 == xmatcopy);
        LIBXSMM_XCOPY(
          LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
          LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL_NOPF, xmatcopy/*0*/, out, in,
          typesize, uldi, uldo, tm, tn, 0, m, 0, n);
      }
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (0 > tid || tid >= nthreads) {
        fprintf(stderr, "LIBXSMM ERROR: the matcopy thread-id or number of threads is incorrect!\n");
      }
      else if (0 == out) {
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


LIBXSMM_API_DEFINITION int libxsmm_matcopy(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch)
{
  return libxsmm_matcopy_thread(out, in, typesize, m, n, ldi, ldo, prefetch, 0/*tid*/, 1/*nthreads*/);
}


LIBXSMM_API_DEFINITION int libxsmm_otrans_thread(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  int tid, int nthreads)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;

  assert(typesize <= 255);
  if (0 != out && 0 != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && n <= ldo &&
    /* use (signed) integer types, but check sanity of input */
    0 <= tid && tid < nthreads)
  {
    LIBXSMM_INIT
    if (out != in) {
      libxsmm_xtransfunction xtrans = 0;
      libxsmm_transpose_descriptor descriptor = { 0 };
      const unsigned int uldi = (unsigned int)ldi, uldo = (unsigned int)ldo;
      const unsigned int size = (unsigned int)(1U * m * n);
      if ((LIBXSMM_TRANS_THRESHOLD) < size) { /* tiled transpose */
        const int tindex = (4 < typesize ? 0 : 1), index = LIBXSMM_MIN(LIBXSMM_SQRT2(size) >> 10, 7);
        libxsmm_blasint m0 = 0, n0 = 0, m1 = m, n1 = n;
        int mtasks;
        descriptor.m = LIBXSMM_MIN(libxsmm_trans_tile[tindex][0/*M*/][index], (unsigned int)m);
        descriptor.n = LIBXSMM_MIN(libxsmm_trans_tile[tindex][1/*N*/][index], (unsigned int)n);
        if (0 != (2 & libxsmm_trans_jit)) { /* JIT'ted transpose permitted? */
          descriptor.typesize = (unsigned char)typesize; descriptor.ldo = uldo;
          /* limit the amount of (unrolled) code by limiting the shape of the kernel */
          if ((LIBXSMM_MAX_M) < descriptor.m) descriptor.m = LIBXSMM_MAX_M;
          if ((LIBXSMM_MAX_N) < descriptor.n) descriptor.n = LIBXSMM_MAX_N;
          xtrans = libxsmm_xtransdispatch(&descriptor);
        }
        mtasks = ((1 < nthreads) ? ((int)((m + descriptor.m - 1) / descriptor.m)) : 1);
        if (1 < mtasks && nthreads <= mtasks) { /* only parallelized over M */
          const int mc = (mtasks + nthreads - 1) / nthreads * descriptor.m;
          m0 = tid * mc; m1 = LIBXSMM_MIN(m0 + mc, m);
        }
        else if (1 < nthreads) {
          const int mc = descriptor.m, ntasks = (nthreads / mtasks);
          const int nc = (((n + ntasks - 1) / ntasks + descriptor.n - 1) / descriptor.n) * descriptor.n;
          const int mtid = tid / ntasks, ntid = tid - mtid * ntasks;
          m0 = mtid * mc; m1 = LIBXSMM_MIN(m0 + mc, m);
          n0 = ntid * nc; n1 = LIBXSMM_MIN(n0 + nc, n);
        }
        assert(((tid + 1) != nthreads) || (m1 == m && n1 == n));
        LIBXSMM_XCOPY(
          LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
          LIBXSMM_TCOPY_KERNEL, LIBXSMM_TCOPY_CALL, xtrans, out, in,
          typesize, uldi, uldo, descriptor.m, descriptor.n, m0, m1, n0, n1);
      }
      else { /* no tiling */
        if (0 != (2 & libxsmm_trans_jit)) { /* JIT'ted transpose permitted? */
          descriptor.typesize = (unsigned char)typesize;
          descriptor.ldo = (unsigned int)ldo;
          descriptor.m = (unsigned int)m;
          descriptor.n = (unsigned int)n;
          xtrans = libxsmm_xtransdispatch(&descriptor);
        }
        if (0 != xtrans) { /* JIT'ted kernel available */
          LIBXSMM_TCOPY_CALL(xtrans, typesize, in, &uldi, out, &uldo);
        }
        else { /* JIT not available */
          LIBXSMM_XCOPY_NONJIT(LIBXSMM_TCOPY_KERNEL, out, in, typesize, uldi, uldo, 0, m, 0, n);
        }
      }
    }
    else if (ldi == ldo) {
      result = libxsmm_itrans(out, typesize, m, n, ldi);
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
      if (0 > tid || tid >= nthreads) {
        fprintf(stderr, "LIBXSMM ERROR: the transpose thread-id or number of threads is incorrect!\n");
      }
      else if (0 == out || 0 == in) {
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


LIBXSMM_API_DEFINITION int libxsmm_otrans(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_otrans_thread(out, in, typesize, m, n, ldi, ldo, 0/*tid*/, 1/*nthreads*/);
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
        fprintf(stderr, "LIBXSMM ERROR: in-place transpose is not fully implemented!\n");
      }
      assert(0/*TODO: proper implementation is pending*/);
      result = EXIT_FAILURE;
    }
    if ((1 < libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM WARNING: in-place transpose is not fully implemented!\n");
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: the transpose input/output cannot be NULL!\n");
    }
    result = EXIT_FAILURE;
  }

  return result;
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
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* /*inout*/, const unsigned int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ld*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* inout, const unsigned int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ld)
{
  assert(0 != typesize && 0 != m);
  libxsmm_itrans(inout, *typesize, *m, *(n ? n : m), *(0 != ld ? ld : m));
}

#endif /*defined(LIBXSMM_BUILD)*/

