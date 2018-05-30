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
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_TRANS_JIT)
# if defined(_WIN32) || defined(__CYGWIN__)
/* only enable matcopy code generation (workaround issue with taking GP registers correctly) */
#   define LIBXSMM_TRANS_JIT 1
# else
#   define LIBXSMM_TRANS_JIT 3
# endif
#endif


LIBXSMM_API_INTERN void libxsmm_trans_init(int archid)
{
  /* setup tile sizes according to CPUID or environment (LIBXSMM_TRANS_M, LIBXSMM_TRANS_N) */
  static unsigned int tile_configs[/*configs*/][2/*DP/SP*/][2/*TILE_M,TILE_N*/][8/*size-range*/] = {
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
  { /* check if JIT-code generation is permitted */
    const char *const env_jit = getenv("LIBXSMM_TRANS_JIT");
    /* determine if JIT-kernels are used (0: none, 1: matcopy, 2: transpose, 3: matcopy+transpose). */
    libxsmm_trans_jit = ((0 == env_jit || 0 == *env_jit) ? (LIBXSMM_TRANS_JIT) : atoi(env_jit));
  }
  { /* load/adjust tile sizes */
    const char *const env_m = getenv("LIBXSMM_TRANS_M"), *const env_n = getenv("LIBXSMM_TRANS_N");
    const int m = ((0 == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((0 == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    int i;
    if (LIBXSMM_X86_AVX512_CORE <= archid) {
      libxsmm_trans_tile = tile_configs[2];
    }
    else if (LIBXSMM_X86_AVX512_MIC <= archid && LIBXSMM_X86_AVX512_CORE > archid) {
      libxsmm_trans_tile = tile_configs[1];
    }
    else {
      libxsmm_trans_tile = tile_configs[0];
    }
    for (i = 0; i < 8; ++i) {
      if (0 < m) libxsmm_trans_tile[0/*DP*/][0/*M*/][i] = libxsmm_trans_tile[1/*SP*/][0/*M*/][i] = m;
      if (0 < n) libxsmm_trans_tile[0/*DP*/][1/*N*/][i] = libxsmm_trans_tile[1/*SP*/][1/*N*/][i] = n;
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_trans_finalize(void)
{
}


LIBXSMM_API int libxsmm_matcopy_thread(void* out, const void* in, unsigned int typesize,
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
    unsigned int tm = (unsigned int)m, tn = (unsigned int)n;
    const int iprefetch = (0 == prefetch ? 0 : *prefetch);
    libxsmm_xmcopyfunction xmatcopy = 0;
    LIBXSMM_INIT /* before leading tile sizes */
    if (1 < nthreads) {
      libxsmm_blasint m0 = 0, n0 = 0, m1 = m, n1 = n;
      const unsigned int size = tm * tn, size2 = LIBXSMM_SQRT2(size);
      const unsigned int indx = LIBXSMM_MIN(size2 >> 10, 7);
      const unsigned int tidx = (4 < typesize ? 0 : 1);
      int mtasks;
      tm = LIBXSMM_MIN(tm, libxsmm_trans_tile[tidx][0/*M*/][indx]);
      tn = LIBXSMM_MIN(tn, libxsmm_trans_tile[tidx][1/*N*/][indx]);
      mtasks = ((1 < nthreads) ? ((int)((m + tm - 1) / tm)) : 1);
      if (1 < mtasks && nthreads <= mtasks) { /* only parallelized over M */
        const int mc = (mtasks + nthreads - 1) / nthreads * tm;
        m0 = tid * mc; m1 = LIBXSMM_MIN(m0 + mc, m);
      }
      else if (1 < nthreads) {
        const int ntasks = nthreads / mtasks, mtid = tid / ntasks, ntid = tid - mtid * ntasks;
        const libxsmm_blasint nc = (((n + ntasks - 1) / ntasks + tn - 1) / tn) * tn;
        const libxsmm_blasint mc = tm;
        m0 = mtid * mc; m1 = LIBXSMM_MIN(m0 + mc, m);
        n0 = ntid * nc; n1 = LIBXSMM_MIN(n0 + nc, n);
      }
      if (0 != (1 & libxsmm_trans_jit) /* libxsmm_trans_jit: JIT'ted matrix-copy permitted? */
        && (1 == typesize || 2 == typesize || 4 == typesize) /* TODO: support multiples */
        /* avoid code-dispatch if task does not need the kernel for inner tiles */
        && tm + m0 <= (unsigned int)(m1 - m0) && tn <= (unsigned int)(n1 - n0)
        /* TODO: investigate issue with Byte-element copy/MT on pre-AVX512 */
        && (1 < typesize || LIBXSMM_X86_AVX2 < libxsmm_target_archid))
      {
        libxsmm_descriptor_blob blob;
        const libxsmm_mcopy_descriptor *const desc = libxsmm_mcopy_descriptor_init(&blob,
          typesize, tm, tn, uldo, uldi, 0 != in ? 0 : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE,
          iprefetch, NULL/*default unroll*/);
        xmatcopy = libxsmm_dispatch_mcopy(desc);
      }
      if (0 != prefetch && 0 != *prefetch) { /* prefetch */
        LIBXSMM_XCOPY(
          LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
          LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL, xmatcopy, out, in,
          typesize, uldi, uldo, tm, tn, m0, m1, n0, n1);
      }
      else { /* no prefetch */
        LIBXSMM_XCOPY(
          LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
          LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL_NOPF, xmatcopy, out, in,
          typesize, uldi, uldo, tm, tn, m0, m1, n0, n1);
      }
    }
    else {
      libxsmm_descriptor_blob blob;
      /* libxsmm_trans_jit: JIT'ted matrix-copy permitted? */
      const libxsmm_mcopy_descriptor *const desc = (0 != (1 & libxsmm_trans_jit) ? libxsmm_mcopy_descriptor_init(&blob,
        typesize, tm, tn, uldo, uldi, 0 != in ? 0 : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE, iprefetch, NULL/*default unroll*/) : 0);
      xmatcopy = libxsmm_dispatch_mcopy(desc);
      assert(0 == tid && 1 == nthreads);
      if (0 != xmatcopy) { /* JIT-kernel available */
        if (0 != prefetch && 0 != *prefetch) { /* prefetch */
          LIBXSMM_MCOPY_CALL(xmatcopy, typesize, in, &uldi, out, &uldo);
        }
        else { /* no prefetch */
          LIBXSMM_MCOPY_CALL_NOPF(xmatcopy, typesize, in, &uldi, out, &uldo);
        }
      }
      else { /* no JIT */
        const unsigned int size = tm * tn, size2 = LIBXSMM_SQRT2(size);
        const unsigned int indx = LIBXSMM_MIN(size2 >> 10, 7);
        const unsigned int tidx = (4 < typesize ? 0 : 1);
        tm = LIBXSMM_MIN(tm, libxsmm_trans_tile[tidx][0/*M*/][indx]);
        tn = LIBXSMM_MIN(tn, libxsmm_trans_tile[tidx][1/*N*/][indx]);
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
        assert(ldi < m || ldo < m);
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the matcopy is/are too small!\n");
      }
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_matcopy(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch)
{
  return libxsmm_matcopy_thread(out, in, typesize, m, n, ldi, ldo, prefetch, 0/*tid*/, 1/*nthreads*/);
}


LIBXSMM_API int libxsmm_otrans_thread(void* out, const void* in, unsigned int typesize,
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
    LIBXSMM_INIT /* before leading tile sizes */
    if (out != in) {
      const unsigned int uldi = (unsigned int)ldi, uldo = (unsigned int)ldo;
      unsigned int tm = (unsigned int)m, tn = (unsigned int)n;
      libxsmm_descriptor_blob blob;
      /* libxsmm_trans_jit: JIT'ted transpose permitted? */
      libxsmm_trans_descriptor* desc = (0 != (2 & libxsmm_trans_jit)
        ? libxsmm_trans_descriptor_init(&blob, typesize, tm, tn, uldo) : 0);
      libxsmm_xtransfunction xtrans = 0;
      if (0 == desc) { /* tiled transpose */
        const unsigned int size = tm * tn, size2 = LIBXSMM_SQRT2(size);
        const unsigned int indx = LIBXSMM_MIN(size2 >> 10, 7);
        const unsigned int tidx = (4 < typesize ? 0 : 1);
        libxsmm_blasint m0 = 0, n0 = 0, m1 = m, n1 = n;
        int mtasks;
        tm = LIBXSMM_MIN(tm, libxsmm_trans_tile[tidx][0/*M*/][indx]);
        tn = LIBXSMM_MIN(tn, libxsmm_trans_tile[tidx][1/*N*/][indx]);
        /* libxsmm_trans_jit: JIT'ted transpose permitted? */
        desc = (0 != (2 & libxsmm_trans_jit) ? libxsmm_trans_descriptor_init(&blob, typesize, tm, tn, uldo) : 0);
        if (0 != desc) { /* limit the amount of (unrolled) code with smaller kernel/tiles */
          desc->m = LIBXSMM_MIN(tm, LIBXSMM_MAX_M); desc->n = LIBXSMM_MIN(tn, LIBXSMM_MAX_N);
          if (0 != (xtrans = libxsmm_dispatch_trans(desc))) {
            tm = desc->m; tn = desc->n;
          }
        }
        mtasks = ((1 < nthreads) ? ((int)((m + tm - 1) / tm)) : 1);
        if (1 < mtasks && nthreads <= mtasks) { /* only parallelized over M */
          const int mc = (mtasks + nthreads - 1) / nthreads * tm;
          m0 = tid * mc; m1 = LIBXSMM_MIN(m0 + mc, m);
        }
        else if (1 < nthreads) {
          const int ntasks = nthreads / mtasks, mtid = tid / ntasks, ntid = tid - mtid * ntasks;
          const libxsmm_blasint nc = (((n + ntasks - 1) / ntasks + tn - 1) / tn) * tn;
          const libxsmm_blasint mc = tm;
          m0 = mtid * mc; m1 = LIBXSMM_MIN(m0 + mc, m);
          n0 = ntid * nc; n1 = LIBXSMM_MIN(n0 + nc, n);
        }
        LIBXSMM_XCOPY(
          LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
          LIBXSMM_TCOPY_KERNEL, LIBXSMM_TCOPY_CALL, xtrans, out, in,
          typesize, uldi, uldo, tm, tn, m0, m1, n0, n1);
      }
      else { /* no tiling */
        if (0 != (xtrans = libxsmm_dispatch_trans(desc))) { /* JIT'ted kernel available */
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


LIBXSMM_API int libxsmm_otrans(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  return libxsmm_otrans_thread(out, in, typesize, m, n, ldi, ldo, 0/*tid*/, 1/*nthreads*/);
}


LIBXSMM_API int libxsmm_itrans(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (0 != inout) {
    LIBXSMM_INIT
    if (m == n) { /* some fall-back; still warned as "not implemented" */
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
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matcopy)(void* out, const void* in, const unsigned int* typesize,
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
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_otrans)(void* out, const void* in, const unsigned int* typesize,
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
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* inout, const unsigned int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ld)
{
  assert(0 != typesize && 0 != m);
  libxsmm_itrans(inout, *typesize, *m, *(n ? n : m), *(0 != ld ? ld : m));
}

#endif /*defined(LIBXSMM_BUILD)*/

