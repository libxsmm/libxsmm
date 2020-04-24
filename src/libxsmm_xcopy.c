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
#include "libxsmm_main.h"

#if !defined(LIBXSMM_XCOPY_JIT)
# if (defined(_WIN32) || defined(__CYGWIN__))
/* only enable matcopy code generation (workaround issue with taking GP registers correctly) */
#   define LIBXSMM_XCOPY_JIT 1
# else
#   define LIBXSMM_XCOPY_JIT 3
# endif
#endif
#if !defined(LIBXSMM_MCOPY_JIT_TINY) && 0
# define LIBXSMM_MCOPY_JIT_TINY
#endif


/* definition of corresponding variables */
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_xcopy_jit);
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_xcopy_taskscale);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_mcopy_prefetch);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_mcopy_mbytes);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_mzero_mbytes);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_tcopy_mbytes);
LIBXSMM_APIVAR_PUBLIC_DEF(float libxsmm_mcopy_nscale);
LIBXSMM_APIVAR_PUBLIC_DEF(float libxsmm_mzero_nscale);
LIBXSMM_APIVAR_PUBLIC_DEF(float libxsmm_tcopy_nscale);


LIBXSMM_API_INTERN void libxsmm_xcopy_init(int archid)
{
  { /* setup tile sizes according to CPUID or environment */
    if (LIBXSMM_X86_AVX512_CORE <= archid) { /* avx-512/core */
      libxsmm_mcopy_prefetch = 0;
      libxsmm_mcopy_mbytes = 1144;
      libxsmm_mcopy_nscale = 0.224f;
      libxsmm_mzero_mbytes = 704;
      libxsmm_mzero_nscale = 0.318f;
      libxsmm_tcopy_mbytes = 16;
      libxsmm_tcopy_nscale = 32.f;
    }
    else if (LIBXSMM_X86_AVX512_MIC <= archid && LIBXSMM_X86_AVX512_CORE > archid) {
      libxsmm_mcopy_prefetch = 1;
      libxsmm_mcopy_mbytes = 1144;
      libxsmm_mcopy_nscale = 0.224f;
      libxsmm_mzero_mbytes = 704;
      libxsmm_mzero_nscale = 0.318f;
      libxsmm_tcopy_mbytes = 16;
      libxsmm_tcopy_nscale = 32.f;
    }
    else { /* avx2 */
      libxsmm_mcopy_prefetch = 0;
      libxsmm_mcopy_mbytes = 1224;
      libxsmm_mcopy_nscale = 0.392f;
      libxsmm_mzero_mbytes = 704;
      libxsmm_mzero_nscale = 0.318f;
      libxsmm_tcopy_mbytes = 16;
      libxsmm_tcopy_nscale = 32.f;
    }
  }
  { /* mcopy: load/adjust tile sizes */
    const char* const env_m = getenv("LIBXSMM_MCOPY_M"), * const env_n = getenv("LIBXSMM_MCOPY_N");
    const int m = ((NULL == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((NULL == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    if (0 < m) libxsmm_mcopy_mbytes = LIBXSMM_MAX(m, 1);
    if (0 < n) libxsmm_mcopy_nscale = ((float)n) / libxsmm_mcopy_mbytes;
    if (1 > (libxsmm_mcopy_nscale * libxsmm_mcopy_mbytes)) {
      const float stretch = 1.f / libxsmm_mcopy_mbytes;
      libxsmm_mcopy_nscale = LIBXSMM_MAX(stretch, libxsmm_mcopy_nscale);
    }
    libxsmm_mcopy_mbytes *= 8; /* measured as if DP */
  }
  { /* mzero: load/adjust tile sizes */
    const char* const env_m = getenv("LIBXSMM_MZERO_M"), * const env_n = getenv("LIBXSMM_MZERO_N");
    const int m = ((NULL == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((NULL == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    if (0 < m) libxsmm_mzero_mbytes = LIBXSMM_MAX(m, 1);
    if (0 < n) libxsmm_mzero_nscale = ((float)n) / libxsmm_mzero_mbytes;
    if (1 > (libxsmm_mzero_nscale * libxsmm_mzero_mbytes)) {
      const float stretch = 1.f / libxsmm_mzero_mbytes;
      libxsmm_mzero_nscale = LIBXSMM_MAX(stretch, libxsmm_mzero_nscale);
    }
    libxsmm_mzero_mbytes *= 8; /* measured as if DP */
  }
  { /* tcopy: load/adjust tile sizes */
    const char* const env_m = getenv("LIBXSMM_TCOPY_M"), * const env_n = getenv("LIBXSMM_TCOPY_N");
    const int m = ((NULL == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((NULL == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    if (0 < m) libxsmm_tcopy_mbytes = LIBXSMM_MAX(m, 1);
    if (0 < n) libxsmm_tcopy_nscale = ((float)n) / libxsmm_tcopy_mbytes;
    if (1 > (libxsmm_tcopy_nscale * libxsmm_tcopy_mbytes)) {
      const float stretch = 1.f / libxsmm_tcopy_mbytes;
      libxsmm_tcopy_nscale = LIBXSMM_MAX(stretch, libxsmm_tcopy_nscale);
    }
    libxsmm_tcopy_mbytes *= 8; /* measured as if DP */
  }
  { /* check if JIT-code generation is permitted */
    const char *const env_jit = getenv("LIBXSMM_XCOPY_JIT");
    /* determine if JIT-kernels are used (0: none, 1: matcopy, 2: transpose, 3: matcopy+transpose). */
    libxsmm_xcopy_jit = ((NULL == env_jit || 0 == *env_jit) ? (LIBXSMM_XCOPY_JIT) : atoi(env_jit));
  }
  { /* determines if OpenMP tasks are used (when available) */
    const char *const env_t = getenv("LIBXSMM_XCOPY_TASKS");
    libxsmm_xcopy_taskscale = ((NULL == env_t || 0 == *env_t)
      ? 0/*disabled*/ : (LIBXSMM_XCOPY_TASKSCALE * atoi(env_t)));
  }
}


LIBXSMM_API_INTERN void libxsmm_xcopy_finalize(void)
{
}


LIBXSMM_API void libxsmm_matcopy_thread_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int tm, unsigned int tn, libxsmm_xmcopyfunction kernel,
  int tid, int nthreads)
{
  const int mtasks = (m + tm - 1) / tm;
  unsigned int m0, m1, n0, n1;

  LIBXSMM_ASSERT_MSG(tid < nthreads && 0 < nthreads, "Invalid task setup");
  LIBXSMM_ASSERT_MSG(tm <= m && tn <= n, "Invalid problem size");
  LIBXSMM_ASSERT_MSG(0 < tm && 0 < tn, "Invalid tile size");
  LIBXSMM_ASSERT_MSG(typesize <= 255, "Invalid type-size");
  LIBXSMM_ASSERT(0 < mtasks);

  if (nthreads <= mtasks) { /* parallelized over M */
    const unsigned int mt = (m + nthreads - 1) / nthreads;
    m0 = LIBXSMM_MIN(tid * mt, m); m1 = LIBXSMM_MIN(m0 + mt, m);
    n0 = 0; n1 = n;
  }
  else { /* parallelized over M and N */
    const int ntasks = nthreads / mtasks;
    const int mtid = tid / ntasks, ntid = tid - mtid * ntasks;
    const unsigned int nt = (((n + ntasks - 1) / ntasks + tn - 1) / tn) * tn;
    m0 = LIBXSMM_MIN(mtid * tm, m); m1 = LIBXSMM_MIN(m0 + tm, m);
    n0 = LIBXSMM_MIN(ntid * nt, n); n1 = LIBXSMM_MIN(n0 + nt, n);
  }

  LIBXSMM_ASSERT_MSG(m0 <= m1 && m1 <= m, "Invalid task size");
  LIBXSMM_ASSERT_MSG(n0 <= n1 && n1 <= n, "Invalid task size");

  if (NULL != in) { /* copy-kernel */
    libxsmm_matcopy_internal(out, in, typesize, ldi, ldo,
      m0, m1, n0, n1, tm, tn, kernel);
  }
  else {
    libxsmm_matzero_internal(out, typesize, ldo,
      m0, m1, n0, n1, tm, tn, kernel);
  }
}


LIBXSMM_API_INTERN void libxsmm_matcopy_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xmcopyfunction kernel)
{
  int prefetch;
  LIBXSMM_ASSERT(NULL != in);
  if (NULL == kernel) {
    prefetch = 0;
  }
  else {
    const libxsmm_descriptor* desc;
    libxsmm_code_pointer code;
    code.xmatcopy = kernel;
    LIBXSMM_EXPECT_NOT(NULL, libxsmm_get_kernel_xinfo(code, &desc, NULL/*code_size*/));
    LIBXSMM_ASSERT(NULL != desc && LIBXSMM_KERNEL_KIND_MCOPY == desc->kind);
    prefetch = desc->mcopy.desc.prefetch;
  }
  if (0 == prefetch) {
    LIBXSMM_XCOPY(LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL, kernel,
      out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1,
      LIBXSMM_XALIGN_MCOPY);
  }
  else {
    LIBXSMM_XCOPY(LIBXSMM_MCOPY_KERNEL, LIBXSMM_MCOPY_CALL_PF, kernel,
      out, in, typesize, ldi, ldo, tm, tn, m0, m1, n0, n1,
      LIBXSMM_XALIGN_MCOPY);
  }
}


LIBXSMM_API_INTERN void libxsmm_matzero_internal(void* out, unsigned int typesize, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xmcopyfunction kernel)
{
  LIBXSMM_XCOPY(LIBXSMM_MZERO_KERNEL, LIBXSMM_MZERO_CALL, kernel,
    out, NULL, typesize, 0, ldo, tm, tn, m0, m1, n0, n1,
    LIBXSMM_XALIGN_MCOPY);
}


LIBXSMM_API void libxsmm_matcopy_thread(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  int tid, int nthreads)
{
  LIBXSMM_INIT
  if (0 < typesize && m <= ldi && m <= ldo && out != in &&
    ((NULL != out && 0 < m && 0 < n) || (0 == m && 0 == n)) &&
    /* use (signed) integer types, but check sanity of input */
    0 <= tid && tid < nthreads)
  {
    if (0 < m && 0 < n) {
      libxsmm_xmcopyfunction kernel = NULL;
      unsigned int tm, tn;
      int prefetch;
      if (NULL != in) {
        prefetch = libxsmm_mcopy_prefetch;
        tm = (libxsmm_mcopy_mbytes + typesize - 1) / typesize;
        tn = (unsigned int)(libxsmm_mcopy_nscale * tm);
      }
      else {
        prefetch = 0;
        tm = (libxsmm_mzero_mbytes + typesize - 1) / typesize;
        tn = (unsigned int)(libxsmm_mzero_nscale * tm);
      }
      if ((unsigned int)m < tm || (unsigned int)n < tn) {
        if (1 == nthreads) {
          tm = (unsigned int)m; tn = (unsigned int)n;
        }
        else {
          const unsigned int tasksize = (((unsigned int)m) * (unsigned int)n) / ((unsigned int)(nthreads * libxsmm_mcopy_nscale));
          const unsigned int nn = libxsmm_isqrt_u32(tasksize);
          const unsigned int mm = (unsigned int)(libxsmm_mcopy_nscale * nn);
          tn = LIBXSMM_CLMP((unsigned int)n, 1, nn);
          tm = LIBXSMM_CLMP((unsigned int)m, 1, mm);
        }
      }
#if !defined(LIBXSMM_MCOPY_JIT_TINY)
      else
#endif
      if (0 != (1 & libxsmm_xcopy_jit)) { /* JIT'ted matrix-copy permitted? */
        libxsmm_descriptor_blob blob;
        kernel = libxsmm_dispatch_mcopy(libxsmm_mcopy_descriptor_init(&blob,
          typesize, tm, tn, (unsigned int)ldo, (unsigned int)ldi,
          NULL != in ? LIBXSMM_MATCOPY_FLAG_DEFAULT : LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE,
          prefetch, NULL/*default unroll*/));
      }
      libxsmm_matcopy_thread_internal(out, in, typesize,
        (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
        tm, tn, kernel, tid, nthreads);
    }
  }
  else {
    static int error_once = 0;
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (0 > tid || tid >= nthreads) {
        fprintf(stderr, "LIBXSMM ERROR: the matrix-copy thread-id or number of threads is incorrect!\n");
      }
      else if (NULL == out) {
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


LIBXSMM_API void libxsmm_matcopy(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  libxsmm_matcopy_thread(out, in, typesize, m, n, ldi, ldo, 0/*tid*/, 1/*nthreads*/);
}


LIBXSMM_API void libxsmm_otrans_thread_internal(void* out, const void* in, unsigned int typesize,
  unsigned int m, unsigned int n, unsigned int ldi, unsigned int ldo,
  unsigned int tm, unsigned int tn, libxsmm_xtransfunction kernel,
  int tid, int nthreads)
{
  const int mtasks = (m + tm - 1) / tm;
  unsigned int m0, m1, n0, n1;

  LIBXSMM_ASSERT_MSG(tid < nthreads && 0 < nthreads, "Invalid task setup");
  LIBXSMM_ASSERT_MSG(tm <= m && tn <= n, "Invalid problem size");
  LIBXSMM_ASSERT_MSG(0 < tm && 0 < tn, "Invalid tile size");
  LIBXSMM_ASSERT_MSG(typesize <= 255, "Invalid type-size");
  LIBXSMM_ASSERT(0 < mtasks);

  if (nthreads <= mtasks) { /* parallelized over M */
    const unsigned int mt = (m + nthreads - 1) / nthreads;
    m0 = LIBXSMM_MIN(tid * mt, m); m1 = LIBXSMM_MIN(m0 + mt, m);
    n0 = 0; n1 = n;
  }
  else { /* parallelized over M and N */
    const int ntasks = nthreads / mtasks;
    const int mtid = tid / ntasks, ntid = tid - mtid * ntasks;
    const unsigned int nt = (((n + ntasks - 1) / ntasks + tn - 1) / tn) * tn;
    m0 = LIBXSMM_MIN(mtid * tm, m); m1 = LIBXSMM_MIN(m0 + tm, m);
    n0 = LIBXSMM_MIN(ntid * nt, n); n1 = LIBXSMM_MIN(n0 + nt, n);
  }

  LIBXSMM_ASSERT_MSG(m0 <= m1 && m1 <= m, "Invalid task size");
  LIBXSMM_ASSERT_MSG(n0 <= n1 && n1 <= n, "Invalid task size");

  libxsmm_otrans_internal(out, in, typesize, ldi, ldo, m0, m1, n0, n1, tm, tn, kernel);
}


LIBXSMM_API_INTERN void libxsmm_otrans_internal(void* out, const void* in,
  unsigned int typesize, unsigned int ldi, unsigned int ldo,
  unsigned int m0, unsigned int m1, unsigned int n0, unsigned int n1,
  unsigned int tm, unsigned int tn, libxsmm_xtransfunction kernel)
{
  LIBXSMM_ASSERT(NULL != in);
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
  if (0 < typesize && m <= ldi && n <= ldo &&
    ((NULL != out && NULL != in && 0 < m && 0 < n) || (0 == m && 0 == n)) &&
    /* use (signed) integer types, but check sanity of input */
    0 <= tid && tid < nthreads)
  {
    if (0 < m && 0 < n) {
      if (out != in) {
        unsigned int tm = (libxsmm_tcopy_mbytes + typesize - 1) / typesize;
        unsigned int tn = (unsigned int)(libxsmm_tcopy_nscale * tm);
        libxsmm_xtransfunction kernel = NULL;
        if ((unsigned int)m < tm || (unsigned int)n < tn) {
          libxsmm_descriptor_blob blob;
          if (1 == nthreads) {
            if (0 != (2 & libxsmm_xcopy_jit) /* JIT'ted transpose permitted? */
              && NULL != (kernel = libxsmm_dispatch_trans( /* JIT-kernel available? */
                libxsmm_trans_descriptor_init(&blob, typesize, (unsigned int)m, (unsigned int)n, (unsigned int)ldo))))
            {
              LIBXSMM_TCOPY_CALL(kernel, typesize, in, ldi, out, ldo);
              return; /* fast path */
            }
            LIBXSMM_ASSERT(NULL == kernel);
            tm = (unsigned int)m; tn = (unsigned int)n;
          }
          else {
            const unsigned int tasksize = (((unsigned int)m) * (unsigned int)n) / ((unsigned int)(nthreads * libxsmm_tcopy_nscale));
            const unsigned int nn = libxsmm_isqrt_u32(tasksize);
            const unsigned int mm = (unsigned int)(libxsmm_tcopy_nscale * nn);
            const libxsmm_trans_descriptor* desc;
            tn = LIBXSMM_CLMP((unsigned int)n, 1, nn);
            tm = LIBXSMM_CLMP((unsigned int)m, 1, mm);
            if (0 != (2 & libxsmm_xcopy_jit) /* JIT'ted transpose permitted? */
              && NULL != (desc = libxsmm_trans_descriptor_init(&blob, typesize, tm, tn, (unsigned int)ldo)))
            {
              kernel = libxsmm_dispatch_trans(desc);
            }
          }
        }
        libxsmm_otrans_thread_internal(out, in, typesize,
          (unsigned int)m, (unsigned int)n, (unsigned int)ldi, (unsigned int)ldo,
          tm, tn, kernel, tid, nthreads);
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
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (0 > tid || tid >= nthreads) {
        fprintf(stderr, "LIBXSMM ERROR: the transpose thread-id or number of threads is incorrect!\n");
      }
      else if (NULL == out || NULL == in) {
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


LIBXSMM_API void libxsmm_otrans(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  libxsmm_otrans_thread(out, in, typesize, m, n, ldi, ldo, 0/*tid*/, 1/*nthreads*/);
}


LIBXSMM_API void libxsmm_itrans(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
{
  static int error_once = 0;
  LIBXSMM_INIT
  if (NULL != inout && 0 < typesize && typesize <= 127 && m <= ld && LIBXSMM_MAX(n, 1) <= ld) {
    const signed char c = (signed char)typesize;
    libxsmm_blasint i, j;
    if (m == n) {
      for (i = 0; i < m; ++i) {
        for (j = 0; j < i; ++j) {
          char *const a = &((char*)inout)[(i*ld+j)*typesize];
          char *const b = &((char*)inout)[(j*ld+i)*typesize];
          signed char k = 0;
          for (; k < c; ++k) LIBXSMM_ISWAP(a[k], b[k]);
        }
      }
    }
    else {
      if ( 0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: in-place transpose is not implemented!\n");
      }
      LIBXSMM_ASSERT_MSG(0, "in-place transpose is not implemented!");
    }
  }
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: unsupported or invalid arguments for in-place transpose!\n");
  }
}


#if defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matcopy)(void* /*out*/, const void* /*in*/, const int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_matcopy)(void* out, const void* in, const int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  LIBXSMM_ASSERT(NULL != typesize && 0 < *typesize && NULL != m);
  ldx = *(NULL != ldi ? ldi : m);
  libxsmm_matcopy(out, in, (unsigned int)*typesize, *m, *(NULL != n ? n : m), ldx, NULL != ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_otrans)(void* /*out*/, const void* /*in*/, const int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ldi*/, const libxsmm_blasint* /*ldo*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_otrans)(void* out, const void* in, const int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo)
{
  libxsmm_blasint ldx;
  LIBXSMM_ASSERT(NULL != typesize && 0 < *typesize && NULL != m);
  ldx = *(NULL != ldi ? ldi : m);
  libxsmm_otrans(out, in, (unsigned int)*typesize, *m, *(NULL != n ? n : m), ldx, NULL != ldo ? *ldo : ldx);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* /*inout*/, const int* /*typesize*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*ld*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_itrans)(void* inout, const int* typesize,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* ld)
{
  LIBXSMM_ASSERT(NULL != typesize && 0 < *typesize && NULL != m);
  libxsmm_itrans(inout, (unsigned int)*typesize, *m, *(NULL != n ? n : m), *(NULL != ld ? ld : m));
}

#endif /*defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/

