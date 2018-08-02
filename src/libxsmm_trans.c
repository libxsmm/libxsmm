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

/* min. tile-size is 3x3 rather than 2x2 to avoid remainder tiles of 1x1 */
#if !defined(LIBXSMM_TRANS_TMIN)
# define LIBXSMM_TRANS_TMIN 3
#endif


LIBXSMM_API_INTERN void libxsmm_trans_init(int archid)
{
  /* setup tile sizes according to CPUID or environment (LIBXSMM_TRANS_M, LIBXSMM_TRANS_N) */
  static libxsmm_blasint config_tm[/*config*/][2/*DP/SP*/] = {
    /* generic (hsw) */ { 2, 2 },
    /* mic (knl/knm) */ { 2, 2 },
    /* core (skx)    */ { 2, 2 }
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
      libxsmm_trans_mtile = config_tm[2];
      libxsmm_trans_tile_stretch = 32.f;
    }
    else if (LIBXSMM_X86_AVX512_MIC <= archid && LIBXSMM_X86_AVX512_CORE > archid) {
      libxsmm_trans_mtile = config_tm[1];
      libxsmm_trans_tile_stretch = 32.f;
    }
    else {
      libxsmm_trans_mtile = config_tm[0];
      libxsmm_trans_tile_stretch = 32.f;
    }
    for (i = 0; i < 2/*DP/SP*/; ++i) {
      if (0 < m) libxsmm_trans_mtile[i] = LIBXSMM_MAX(m, LIBXSMM_TRANS_TMIN);
      if (0 < n) libxsmm_trans_tile_stretch = ((float)n) / libxsmm_trans_mtile[i];
      if (LIBXSMM_TRANS_TMIN > (libxsmm_trans_tile_stretch * libxsmm_trans_mtile[i])) {
        const float stretch = ((float)(LIBXSMM_TRANS_TMIN)) / libxsmm_trans_mtile[i];
        libxsmm_trans_tile_stretch = LIBXSMM_MAX(stretch, libxsmm_trans_tile_stretch);
      }
    }
  }
  { /* determines if OpenMP tasks are used (when available) */
    const char *const env_t = getenv("LIBXSMM_TRANS_TASKS");
    libxsmm_trans_taskscale = ((0 == env_t || 0 == *env_t)
      ? 0/*disabled*/ : (LIBXSMM_TRANS_TASKSCALE * atoi(env_t)));
  }
}


LIBXSMM_API_INTERN void libxsmm_trans_finalize(void)
{
}


LIBXSMM_API void libxsmm_matcopy_internal(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo, const int* prefetch,
  libxsmm_blasint tm, libxsmm_blasint tn, libxsmm_xmcopyfunction kernel,
  int tid, int nthreads)
{
  const int mtasks = (m + tm - 1) / tm;
  libxsmm_blasint m0, m1, n0, n1;

  LIBXSMM_ASSERT_MSG(tid < nthreads && 0 < nthreads, "Invalid task setup!");
  LIBXSMM_ASSERT_MSG(tm <= m && tn <= n, "Invalid problem size!");
  LIBXSMM_ASSERT_MSG(typesize <= 255, "Invalid type-size!");

  if (nthreads <= mtasks) { /* parallelized over M */
    const libxsmm_blasint mt = (m + nthreads - 1) / nthreads;
    m0 = LIBXSMM_MIN(tid * mt, m); m1 = LIBXSMM_MIN(m0 + mt, m);
    n0 = 0; n1 = n;
  }
  else { /* parallelized over M and N */
    const int ntasks = nthreads / mtasks;
    const int mtid = tid / ntasks, ntid = tid - mtid * ntasks;
    const libxsmm_blasint nt = (((n + ntasks - 1) / ntasks + tn - 1) / tn) * tn;
    m0 = LIBXSMM_MIN(mtid * tm, m); m1 = LIBXSMM_MIN(m0 + tm, m);
    n0 = LIBXSMM_MIN(ntid * nt, n); n1 = LIBXSMM_MIN(n0 + nt, n);
  }

  LIBXSMM_ASSERT_MSG(m0 <= m1 && m1 <= m, "Invalid task size!");
  LIBXSMM_ASSERT_MSG(n0 <= n1 && n1 <= n, "Invalid task size!");

  if (0 != prefetch && 0 != *prefetch) { /* prefetch */
    LIBXSMM_XCOPY(LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL, kernel,
      out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1,
      LIBXSMM_XALIGN_MCOPY);
  }
  else { /* no prefetch */
    LIBXSMM_XCOPY(LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL_NOPF, kernel,
      out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1,
      LIBXSMM_XALIGN_MCOPY);
  }
}


LIBXSMM_API void libxsmm_matcopy_thread(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch, int tid, int nthreads)
{
  LIBXSMM_INIT
#if defined(LIBXSMM_TRANS_CHECK)
  if (0 != out && out != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && m <= ldo &&
    /* use (signed) integer types, but check sanity of input */
    0 <= tid && tid < nthreads)
#endif
  {
    libxsmm_blasint tm = libxsmm_trans_mtile[4 < typesize ? 0 : 1];
    libxsmm_blasint tn = (libxsmm_blasint)(libxsmm_trans_tile_stretch * tm);
    libxsmm_xmcopyfunction kernel = NULL;
    if (m < tm || n < tn) {
      if (1 < nthreads) {
        const unsigned int tasksize = (((unsigned int)m) * n) / ((unsigned int)(nthreads * libxsmm_trans_tile_stretch));
        const libxsmm_blasint nn = (libxsmm_blasint)libxsmm_isqrt_u32(tasksize);
        const libxsmm_blasint mm = (libxsmm_blasint)(libxsmm_trans_tile_stretch * nn);
        tn = LIBXSMM_MIN(nn, n); tm = LIBXSMM_CLMP(m, 1, mm);
      }
      else {
        tm = m; tn = n;
      }
    }
    else {
      const int iprefetch = (0 == prefetch ? 0 : *prefetch);
      const libxsmm_mcopy_descriptor* desc;
      libxsmm_descriptor_blob blob;
      if (0 != (1 & libxsmm_trans_jit) /* JIT'ted matrix-copy permitted? */
        && NULL != (desc = libxsmm_mcopy_descriptor_init(&blob, typesize,
        (unsigned int)tm, (unsigned int)tn, (unsigned int)ldo, (unsigned int)ldi,
          0 != in ? 0 : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE, iprefetch, NULL/*default unroll*/)))
      {
        kernel = libxsmm_dispatch_mcopy(desc);
      }
    }
    libxsmm_matcopy_internal(out, in, typesize, m, n, ldi, ldo, prefetch, tm, tn, kernel, tid, nthreads);
  }
#if defined(LIBXSMM_TRANS_CHECK)
  else {
    static int error_once = 0;
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
        LIBXSMM_ASSERT(ldi < m || ldo < m);
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the matcopy is/are too small!\n");
      }
    }
  }
#endif
}


LIBXSMM_API void libxsmm_matcopy(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch)
{
  libxsmm_matcopy_thread(out, in, typesize, m, n, ldi, ldo, prefetch, 0/*tid*/, 1/*nthreads*/);
}


LIBXSMM_API void libxsmm_otrans_thread_internal(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  libxsmm_blasint tm, libxsmm_blasint tn, libxsmm_xtransfunction kernel,
  int tid, int nthreads)
{
  const int mtasks = (m + tm - 1) / tm;
  libxsmm_blasint m0, m1, n0, n1;

  LIBXSMM_ASSERT_MSG(tid < nthreads && 0 < nthreads, "Invalid task setup!");
  LIBXSMM_ASSERT_MSG(tm <= m && tn <= n, "Invalid problem size!");
  LIBXSMM_ASSERT_MSG(typesize <= 255, "Invalid type-size!");

  if (nthreads <= mtasks) { /* parallelized over M */
    const libxsmm_blasint mt = (m + nthreads - 1) / nthreads;
    m0 = LIBXSMM_MIN(tid * mt, m); m1 = LIBXSMM_MIN(m0 + mt, m);
    n0 = 0; n1 = n;
  }
  else { /* parallelized over M and N */
    const int ntasks = nthreads / mtasks;
    const int mtid = tid / ntasks, ntid = tid - mtid * ntasks;
    const libxsmm_blasint nt = (((n + ntasks - 1) / ntasks + tn - 1) / tn) * tn;
    m0 = LIBXSMM_MIN(mtid * tm, m); m1 = LIBXSMM_MIN(m0 + tm, m);
    n0 = LIBXSMM_MIN(ntid * nt, n); n1 = LIBXSMM_MIN(n0 + nt, n);
  }

  LIBXSMM_ASSERT_MSG(m0 <= m1 && m1 <= m, "Invalid task size!");
  LIBXSMM_ASSERT_MSG(n0 <= n1 && n1 <= n, "Invalid task size!");

  libxsmm_otrans_internal(out, in, typesize, ldi, ldo, m0, m1, n0, n1, tm, tn, kernel);
}


LIBXSMM_API_INTERN void libxsmm_otrans_internal(void* out, const void* in,
  unsigned int typesize, libxsmm_blasint ldi, libxsmm_blasint ldo,
  libxsmm_blasint m0, libxsmm_blasint m1, libxsmm_blasint n0, libxsmm_blasint n1,
  libxsmm_blasint tm, libxsmm_blasint tn, libxsmm_xtransfunction kernel)
{
  LIBXSMM_XCOPY(LIBXSMM_TCOPY_KERNEL, LIBXSMM_TCOPY_CALL, kernel,
    out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1,
    LIBXSMM_XALIGN_TCOPY);
}


LIBXSMM_API void libxsmm_otrans_thread(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  int tid, int nthreads)
{
  static int error_once = 0;
  LIBXSMM_INIT
#if defined(LIBXSMM_TRANS_CHECK)
  if (0 != out && 0 != in && 0 < typesize && 0 < m && 0 < n && m <= ldi && n <= ldo &&
    /* use (signed) integer types, but check sanity of input */
    0 <= tid && tid < nthreads)
#endif
  {
    if (out != in) {
      libxsmm_blasint tm = libxsmm_trans_mtile[4 < typesize ? 0 : 1];
      libxsmm_blasint tn = (libxsmm_blasint)(libxsmm_trans_tile_stretch * tm);
      libxsmm_xtransfunction kernel = NULL;
      if (m < tm || n < tn) {
        const libxsmm_trans_descriptor* desc;
        libxsmm_descriptor_blob blob;
        if (1 < nthreads) {
          const unsigned int tasksize = (((unsigned int)m) * n) / ((unsigned int)(nthreads * libxsmm_trans_tile_stretch));
          const libxsmm_blasint nn = (libxsmm_blasint)libxsmm_isqrt_u32(tasksize);
          const libxsmm_blasint mm = (libxsmm_blasint)(libxsmm_trans_tile_stretch * nn);
          tn = LIBXSMM_MIN(nn, n); tm = LIBXSMM_CLMP(m, 1, mm);
          if (0 != (2 & libxsmm_trans_jit) /* JIT'ted transpose permitted? */
            && NULL != (desc = libxsmm_trans_descriptor_init(&blob, typesize, (unsigned int)tm, (unsigned int)tn, (unsigned int)ldo)))
          {
            kernel = libxsmm_dispatch_trans(desc);
          }
        }
        else {
          if (0 != (2 & libxsmm_trans_jit) /* JIT'ted transpose permitted? */
            && NULL != (desc = libxsmm_trans_descriptor_init(&blob, typesize, (unsigned int)m, (unsigned int)n, (unsigned int)ldo))
            && NULL != (kernel = libxsmm_dispatch_trans(desc))) /* JIT-kernel available */
          {
            LIBXSMM_TCOPY_CALL(kernel, typesize, in, ldi, out, ldo);
            return; /* fast path */
          }
          tm = m; tn = n;
        }
      }
      libxsmm_otrans_thread_internal(out, in, typesize, m, n, ldi, ldo, tm, tn, kernel, tid, nthreads);
    }
    else if (ldi == ldo) {
      libxsmm_itrans(out, typesize, m, n, ldi);
    }
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: output and input of the transpose must be different!\n");
    }
  }
#if defined(LIBXSMM_TRANS_CHECK)
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
        LIBXSMM_ASSERT(ldi < m || ldo < n);
        fprintf(stderr, "LIBXSMM ERROR: the leading dimension(s) of the transpose is/are too small!\n");
      }
    }
  }
#endif
}


LIBXSMM_API void libxsmm_otrans(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  libxsmm_otrans_thread(out, in, typesize, m, n, ldi, ldo, 0/*tid*/, 1/*nthreads*/);
}


LIBXSMM_API void libxsmm_itrans(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
#if defined(LIBXSMM_TRANS_CHECK)
  static int error_once = 0;
  LIBXSMM_INIT
  if (0 != inout)
#else
  LIBXSMM_UNUSED(n);
  LIBXSMM_INIT
#endif
  {
#if defined(LIBXSMM_TRANS_CHECK)
    if (m == n) /* some fall-back; still warned as "not implemented" */
#endif
    {
      libxsmm_blasint i, j;
      for (i = 0; i < m; ++i) {
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
#if defined(LIBXSMM_TRANS_CHECK)
      if ((1 < libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM WARNING: in-place transpose is not fully implemented!\n");
      }
#endif
    }
#if defined(LIBXSMM_TRANS_CHECK)
    else {
      if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: in-place transpose is not fully implemented!\n");
      }
      LIBXSMM_ASSERT(0/*TODO: proper implementation is pending*/);
    }
#endif
  }
#if defined(LIBXSMM_TRANS_CHECK)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: the transpose input/output cannot be NULL!\n");
  }
#endif
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
  LIBXSMM_ASSERT(0 != typesize && 0 != m);
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
  LIBXSMM_ASSERT(0 != typesize && 0 != m);
  ldx = *(0 != ldi ? ldi : m);
  libxsmm_otrans(out, in, *typesize, *m, *(n ? n : m), ldx, ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* /*inout*/, const unsigned int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ld*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* inout, const unsigned int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ld)
{
  LIBXSMM_ASSERT(0 != typesize && 0 != m);
  libxsmm_itrans(inout, *typesize, *m, *(n ? n : m), *(0 != ld ? ld : m));
}

#endif /*defined(LIBXSMM_BUILD)*/

