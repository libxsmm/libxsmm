/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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
#include "libxsmm_gemm.h"
#include "libxsmm_trans.h"
#include "libxsmm_hash.h"
#include <libxsmm_mhd.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_GEMM_NOJIT_TRANS) && \
  /* TODO: fully support calling convention */ \
  (defined(_WIN32) || defined(__CYGWIN__))
# define LIBXSMM_GEMM_NOJIT_TRANS
#endif
#if !defined(LIBXSMM_GEMM_MSIZE_MAX)
# define LIBXSMM_GEMM_MSIZE_MAX 48
#endif
#if !defined(LIBXSMM_GEMM_KPARALLEL) && 0
# define LIBXSMM_GEMM_KPARALLEL
#endif
#if !defined(LIBXSMM_GEMM_BATCHSIZE)
# define LIBXSMM_GEMM_BATCHSIZE 1024
#endif
#if !defined(LIBXSMM_GEMM_BATCHGRAIN)
# define LIBXSMM_GEMM_BATCHGRAIN 128
#endif

#if !defined(LIBXSMM_NO_SYNC) /** Locks for the batch interface (duplicated C indexes). */
# define LIBXSMM_GEMM_LOCKIDX(IDX, NPOT) LIBXSMM_MOD2(LIBXSMM_CONCATENATE(libxsmm_crc32_u,LIBXSMM_BLASINT_NBITS)(2507/*seed*/, IDX), NPOT)
# define LIBXSMM_GEMM_LOCKPTR(PTR, NPOT) LIBXSMM_MOD2(libxsmm_crc32_u64(1975/*seed*/, (uintptr_t)(PTR)), NPOT)
# if !defined(LIBXSMM_GEMM_MAXNLOCKS)
#   define LIBXSMM_GEMM_MAXNLOCKS 1024
# endif
# if !defined(LIBXSMM_GEMM_LOCKFWD)
#   define LIBXSMM_GEMM_LOCKFWD
# endif
# if LIBXSMM_LOCK_TYPE_ISPOD(LIBXSMM_GEMM_LOCK)
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE internal_gemm_locktype {
  char pad[LIBXSMM_CACHELINE];
  LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) state;
} internal_gemm_locktype;

# else
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE internal_gemm_locktype {
  LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) state;
} internal_gemm_locktype;
# endif
LIBXSMM_APIVAR(internal_gemm_locktype internal_gemm_lock[LIBXSMM_GEMM_MAXNLOCKS]);
LIBXSMM_APIVAR(unsigned int internal_gemm_nlocks); /* populated number of locks */
#endif

/** Prefetch strategy for tiled GEMM. */
LIBXSMM_APIVAR(libxsmm_gemm_prefetch_type internal_gemm_tiled_prefetch);
/** Table of M-extents per type-size (tile shape). */
LIBXSMM_APIVAR(unsigned int internal_gemm_vwidth);
/** Table of M-extents per type-size (tile shape). */
LIBXSMM_APIVAR(float internal_gemm_nstretch);
/** Table of M-extents per type-size (tile shape). */
LIBXSMM_APIVAR(float internal_gemm_kstretch);


LIBXSMM_API LIBXSMM_GEMM_WEAK libxsmm_dgemm_function libxsmm_original_dgemm(void)
{
  static /*volatile*/ libxsmm_dgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(double, original);
  LIBXSMM_ASSERT(0 != original);
  return original;
}


LIBXSMM_API LIBXSMM_GEMM_WEAK libxsmm_sgemm_function libxsmm_original_sgemm(void)
{
  static /*volatile*/ libxsmm_sgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(float, original);
  LIBXSMM_ASSERT(0 != original);
  return original;
}


LIBXSMM_API_INTERN void libxsmm_gemm_init(int archid)
{
  LIBXSMM_LOCK_ATTR_TYPE(LIBXSMM_GEMM_LOCK) attr;
  unsigned int i;

  LIBXSMM_LOCK_ATTR_INIT(LIBXSMM_GEMM_LOCK, &attr);
  { /* setup prefetch strategy for tiled GEMMs */
    const char *const env_p = getenv("LIBXSMM_TGEMM_PREFETCH");
    const libxsmm_gemm_prefetch_type tiled_prefetch_default = LIBXSMM_GEMM_PREFETCH_AL2_AHEAD;
    const int uid = ((0 == env_p || 0 == *env_p) ? LIBXSMM_PREFETCH_AUTO/*default*/ : atoi(env_p));
    internal_gemm_tiled_prefetch = (0 <= uid ? libxsmm_gemm_uid2prefetch(uid) : tiled_prefetch_default);
  }
#if !defined(LIBXSMM_NO_SYNC)
  { /* initialize locks for the batch interface */
    const char *const env_locks = getenv("LIBXSMM_GEMM_NLOCKS");
    const int nlocks = ((0 == env_locks || 0 == *env_locks) ? -1/*default*/ : atoi(env_locks));
    internal_gemm_nlocks = LIBXSMM_UP2POT(0 > nlocks ? (LIBXSMM_GEMM_MAXNLOCKS) : LIBXSMM_MIN(nlocks, LIBXSMM_GEMM_MAXNLOCKS));
    for (i = 0; i < internal_gemm_nlocks; ++i) LIBXSMM_LOCK_INIT(LIBXSMM_GEMM_LOCK, &internal_gemm_lock[i].state, &attr);
  }
#endif
#if defined(LIBXSMM_GEMM_MMBATCH)
  { const char *const env_w = getenv("LIBXSMM_GEMM_WRAP");
    /* intercepted GEMMs (1: sequential and non-tiled, 2: parallelized and tiled) */
    libxsmm_gemm_wrap = ((0 == env_w || 0 == *env_w) ? (LIBXSMM_WRAP) : atoi(env_w));
    if (0 != libxsmm_gemm_wrap) {
      const char *const env_b = getenv("LIBXSMM_GEMM_BATCHSIZE");
      const unsigned int batchsize = ((0 == env_b || 0 == *env_b || 0 >= atoi(env_b)) ? (LIBXSMM_GEMM_BATCHSIZE) : atoi(env_b));
      void *const p = &libxsmm_gemm_batcharray;
      const void *const extra = 0;
      /* draw default/non-scratch memory, but utilize the scratch memory allocator */
      LIBXSMM_ASSERT(1 < (LIBXSMM_GEMM_BATCHSCALE));
      if (EXIT_SUCCESS == libxsmm_xmalloc((void**)p,
        (size_t)(LIBXSMM_GEMM_BATCHSCALE) * sizeof(libxsmm_gemm_batchitem) * batchsize,
        0, LIBXSMM_MALLOC_FLAG_SCRATCH, &extra, sizeof(extra)))
      {
        const char *const env_g = getenv("LIBXSMM_GEMM_BATCHGRAIN");
        const unsigned int batchgrain = ((0 == env_g || 0 == *env_g || 0 >= atoi(env_g)) ? (LIBXSMM_GEMM_BATCHGRAIN) : atoi(env_g));
        LIBXSMM_LOCK_INIT(LIBXSMM_GEMM_LOCK, &libxsmm_gemm_batchlock, &attr);
        libxsmm_gemm_batchgrain = batchgrain;
        libxsmm_gemm_batchsize = batchsize;
      }
      if (((3 <= libxsmm_verbosity && INT_MAX != libxsmm_verbosity) || 0 > libxsmm_verbosity)
        && (0 == env_w || 0 == *env_w))
      { /* enable auto-batch statistic */
        libxsmm_gemm_batchdesc.flags = LIBXSMM_MMBATCH_FLAG_STATISTIC;
      }
    }
  }
#endif
  if (LIBXSMM_X86_AVX512_CORE <= archid) {
    internal_gemm_vwidth = 64;
    internal_gemm_nstretch = 1;
    internal_gemm_kstretch = 3;
  }
  else if (LIBXSMM_X86_AVX512_MIC <= archid) {
    internal_gemm_vwidth = 64;
    internal_gemm_nstretch = 1;
    internal_gemm_kstretch = 3;
  }
  else if (LIBXSMM_X86_AVX2 <= archid) {
    internal_gemm_vwidth = 32;
    internal_gemm_nstretch = 5;
    internal_gemm_kstretch = 2;
  }
  else if (LIBXSMM_X86_AVX <= archid) {
    internal_gemm_vwidth = 32;
    internal_gemm_nstretch = 5;
    internal_gemm_kstretch = 2;
  }
  else {
    internal_gemm_vwidth = 16;
    internal_gemm_nstretch = 7;
    internal_gemm_kstretch = 2;
  }
  { /* setup tile sizes according to environment (LIBXSMM_TGEMM_M, LIBXSMM_TGEMM_N, LIBXSMM_TGEMM_K) */
    const char *const env_m = getenv("LIBXSMM_TGEMM_M"), *const env_n = getenv("LIBXSMM_TGEMM_N"), *const env_k = getenv("LIBXSMM_TGEMM_K");
    const int m = ((0 == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((0 == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    const int k = ((0 == env_k || 0 == *env_k) ? 0 : atoi(env_k));
    if (0 < m) {
      if (0 < n) internal_gemm_nstretch = ((float)n) / m;
      if (0 < k) internal_gemm_kstretch = ((float)k) / m;
    }
  }
  { /* setup tile sizes according to environment (LIBXSMM_TGEMM_NS, LIBXSMM_TGEMM_KS) */
    const char *const env_ns = getenv("LIBXSMM_TGEMM_NS"), *const env_ks = getenv("LIBXSMM_TGEMM_KS");
    const double ns = ((0 == env_ns || 0 == *env_ns) ? 0 : atof(env_ns));
    const double ks = ((0 == env_ks || 0 == *env_ks) ? 0 : atof(env_ks));
    if (0 < ns) internal_gemm_nstretch = (float)LIBXSMM_MIN(24, ns);
    if (0 < ks) internal_gemm_kstretch = (float)LIBXSMM_MIN(24, ks);
  }
  { /* determines if OpenMP tasks are used (when available) */
    const char *const env_t = getenv("LIBXSMM_GEMM_TASKS");
    libxsmm_gemm_taskscale = ((0 == env_t || 0 == *env_t)
      ? 0/*disabled*/ : (LIBXSMM_GEMM_TASKSCALE * atoi(env_t)));
  }
  LIBXSMM_LOCK_ATTR_DESTROY(LIBXSMM_GEMM_LOCK, &attr);
}


LIBXSMM_API_INTERN void libxsmm_gemm_finalize(void)
{
#if !defined(LIBXSMM_NO_SYNC)
  unsigned int i; for (i = 0; i < internal_gemm_nlocks; ++i) LIBXSMM_LOCK_DESTROY(LIBXSMM_GEMM_LOCK, &internal_gemm_lock[i].state);
#endif
#if defined(LIBXSMM_GEMM_MMBATCH)
  if (0 != libxsmm_gemm_batcharray) {
    void* extra = 0;
    if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(libxsmm_gemm_batcharray, NULL/*size*/, NULL/*flags*/, &extra) && 0 != extra) {
      const libxsmm_mmbatch_flush_function flush = *(libxsmm_mmbatch_flush_function*)extra;
      if (0 != flush) flush();
    }
    libxsmm_free(libxsmm_gemm_batcharray);
    LIBXSMM_LOCK_DESTROY(LIBXSMM_GEMM_LOCK, &libxsmm_gemm_batchlock);
  }
#endif
}


LIBXSMM_API_INTERN unsigned char libxsmm_gemm_typesize(libxsmm_gemm_precision precision)
{
  return libxsmm_typesize((libxsmm_datatype)precision);
}


LIBXSMM_API_INTERN int libxsmm_gemm_dvalue(libxsmm_gemm_precision precision, const void* value, double* dvalue)
{
  return libxsmm_dvalue((libxsmm_datatype)precision, value, dvalue);
}


LIBXSMM_API_INTERN int libxsmm_gemm_cast(libxsmm_gemm_precision precision, double dvalue, void* value)
{
  return libxsmm_cast((libxsmm_datatype)precision, dvalue, value);
}


LIBXSMM_API_INLINE libxsmm_gemm_prefetch_type internal_get_gemm_prefetch(int prefetch)
{
  const int result = (0 > prefetch ? ((int)libxsmm_gemm_auto_prefetch_default) : prefetch);
  LIBXSMM_ASSERT_MSG(0 <= result, "LIBXSMM_PREFETCH_AUTO is not translated!");
  return (libxsmm_gemm_prefetch_type)result;
}


LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_xprefetch(const int* prefetch)
{
  LIBXSMM_INIT /* load configuration */
  return internal_get_gemm_prefetch(0 == prefetch ? ((int)libxsmm_gemm_auto_prefetch) : *prefetch);
}


LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_prefetch(int prefetch)
{
  LIBXSMM_INIT /* load configuration */
  return internal_get_gemm_prefetch(prefetch);
}


LIBXSMM_API_INTERN int libxsmm_gemm_prefetch2uid(libxsmm_gemm_prefetch_type prefetch)
{
  switch (prefetch) {
    case LIBXSMM_GEMM_PREFETCH_SIGONLY:            return 2;
    case LIBXSMM_GEMM_PREFETCH_BL2_VIA_C:          return 3;
    case LIBXSMM_GEMM_PREFETCH_AL2_AHEAD:          return 4;
    case LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD: return 5;
    case LIBXSMM_GEMM_PREFETCH_AL2:                return 6;
    case LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C:       return 7;
    case LIBXSMM_GEMM_PREFETCH_AL2_JPST:           return 8;
    case LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_JPST:  return 9;
    /*case LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C:    return 10;*/
    case LIBXSMM_GEMM_PREFETCH_AL1:                return 10;
    case LIBXSMM_GEMM_PREFETCH_BL1:                return 11;
    case LIBXSMM_GEMM_PREFETCH_CL1:                return 12;
    case LIBXSMM_GEMM_PREFETCH_AL1_BL1:            return 13;
    case LIBXSMM_GEMM_PREFETCH_BL1_CL1:            return 14;
    case LIBXSMM_GEMM_PREFETCH_AL1_CL1:            return 15;
    case LIBXSMM_GEMM_PREFETCH_AL1_BL1_CL1:        return 16;
    default: {
      LIBXSMM_ASSERT(LIBXSMM_GEMM_PREFETCH_NONE == prefetch);
      return 0;
    }
  }
}


LIBXSMM_API_INTERN libxsmm_gemm_prefetch_type libxsmm_gemm_uid2prefetch(int uid)
{
  switch (uid) {
    case  1: return LIBXSMM_GEMM_PREFETCH_NONE;                /* nopf */
    case  2: return LIBXSMM_GEMM_PREFETCH_SIGONLY;             /* pfsigonly */
    case  3: return LIBXSMM_GEMM_PREFETCH_BL2_VIA_C;           /* BL2viaC */
    case  4: return LIBXSMM_GEMM_PREFETCH_AL2_AHEAD;           /* curAL2 */
    case  5: return LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD;  /* curAL2_BL2viaC */
    case  6: return LIBXSMM_GEMM_PREFETCH_AL2;                 /* AL2 */
    case  7: return LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C;        /* AL2_BL2viaC */
    case  8: return LIBXSMM_GEMM_PREFETCH_AL2_JPST;            /* AL2jpst */
    case  9: return LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_JPST;   /* AL2jpst_BL2viaC */
    /*case 10: return LIBXSMM_GEMM_PREFETCH_AL2CL2BL2_VIA_C;*/     /* AL2_BL2viaC_CL2 */
    case 10: return LIBXSMM_GEMM_PREFETCH_AL1;
    case 11: return LIBXSMM_GEMM_PREFETCH_BL1;
    case 12: return LIBXSMM_GEMM_PREFETCH_CL1;
    case 13: return LIBXSMM_GEMM_PREFETCH_AL1_BL1;
    case 14: return LIBXSMM_GEMM_PREFETCH_BL1_CL1;
    case 15: return LIBXSMM_GEMM_PREFETCH_AL1_CL1;
    case 16: return LIBXSMM_GEMM_PREFETCH_AL1_BL1_CL1;
    default: {
      if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
        static int error_once = 0;
        if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
          fprintf(stderr, "LIBXSMM WARNING: invalid prefetch strategy requested!\n");
        }
      }
      return LIBXSMM_GEMM_PREFETCH_NONE;
    }
  }
}


LIBXSMM_API void libxsmm_gemm_print(void* ostream,
  libxsmm_gemm_precision precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  libxsmm_gemm_print2(ostream, precision, precision, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API void libxsmm_gemm_print2(void* ostream,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m);
  const char ctransa = (char)(0 != transa ? (*transa) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'n' : 't'));
  const char ctransb = (char)(0 != transb ? (*transb) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'n' : 't'));
  const libxsmm_blasint ilda = (NULL != lda ? *lda : (('n' == ctransa || 'N' == ctransa) ? *m : kk));
  const libxsmm_blasint ildb = (NULL != ldb ? *ldb : (('n' == ctransb || 'N' == ctransb) ? kk : nn));
  const libxsmm_blasint ildc = *(NULL != ldc ? ldc : m);
  libxsmm_mhd_elemtype mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_CHAR;
  char string_a[128], string_b[128], typeprefix = 0;

  switch (iprec) {
    case LIBXSMM_GEMM_PRECISION_F64: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", 0 != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", 0 != beta  ? *((const double*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F64;
      typeprefix = 'd';
    } break;
    case LIBXSMM_GEMM_PRECISION_F32: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", 0 != alpha ? *((const float*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", 0 != beta  ? *((const float*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F32;
      typeprefix = 's';
    } break;
    default: if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      static int error_once = 0;
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) { /* TODO: support I16, etc. */
        fprintf(stderr, "LIBXSMM ERROR: unsupported data-type requested!\n");
      }
    }
  }

  if (0 != typeprefix) {
    if (0 != ostream) { /* print information about GEMM call */
      if (0 != a && 0 != b && 0 != c) {
        fprintf((FILE*)ostream, "%cgemm('%c', '%c', %" PRIuPTR "/*m*/, %" PRIuPTR "/*n*/, %" PRIuPTR "/*k*/,\n"
                                "  %s/*alpha*/, %p/*a*/, %" PRIuPTR "/*lda*/,\n"
                                "              %p/*b*/, %" PRIuPTR "/*ldb*/,\n"
                                "   %s/*beta*/, %p/*c*/, %" PRIuPTR "/*ldc*/)",
          typeprefix, ctransa, ctransb, (uintptr_t)*m, (uintptr_t)nn, (uintptr_t)kk,
          string_a, a, (uintptr_t)ilda, b, (uintptr_t)ildb, string_b, c, (uintptr_t)ildc);
      }
      else {
        fprintf((FILE*)ostream, "%cgemm(trans=%c%c mnk=%" PRIuPTR ",%" PRIuPTR ",%" PRIuPTR
                                                 " ldx=%" PRIuPTR ",%" PRIuPTR ",%" PRIuPTR " a,b=%s,%s)",
          typeprefix, ctransa, ctransb, (uintptr_t)*m, (uintptr_t)nn, (uintptr_t)kk,
          (uintptr_t)ilda, (uintptr_t)ildb, (uintptr_t)ildc, string_a, string_b);
      }
    }
    else { /* dump A, B, and C matrices into MHD files */
      char extension_header[256];
      size_t data_size[2], size[2];

      if (0 != a) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "TRANS = %c\nALPHA = %s", ctransa, string_a);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_a_%p.mhd", a);
        data_size[0] = (size_t)ilda; data_size[1] = (size_t)kk; size[0] = (size_t)(*m); size[1] = (size_t)kk;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, a, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
      if (0 != b) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "\nTRANS = %c", ctransb);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_b_%p.mhd", b);
        data_size[0] = (size_t)ildb; data_size[1] = (size_t)nn; size[0] = (size_t)kk; size[1] = (size_t)nn;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, b, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
      if (0 != c) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "BETA = %s", string_b);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_c_%p.mhd", c);
        data_size[0] = (size_t)ildc; data_size[1] = (size_t)nn; size[0] = (size_t)(*m); size[1] = (size_t)nn;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, c, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
    }
  }
}


LIBXSMM_API void libxsmm_gemm_dprint(
  void* ostream, libxsmm_gemm_precision precision, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, double dalpha, const void* a, libxsmm_blasint lda,
  const void* b, libxsmm_blasint ldb, double dbeta, void* c, libxsmm_blasint ldc)
{
  libxsmm_gemm_dprint2(ostream, precision, precision, transa, transb, m, n, k, dalpha, a, lda, b, ldb, dbeta, c, ldc);
}


LIBXSMM_API void libxsmm_gemm_dprint2(
  void* ostream, libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, double dalpha, const void* a, libxsmm_blasint lda,
  const void* b, libxsmm_blasint ldb, double dbeta, void* c, libxsmm_blasint ldc)
{
  switch (iprec) {
    case LIBXSMM_GEMM_PRECISION_F64: {
      libxsmm_gemm_print2(ostream, LIBXSMM_GEMM_PRECISION_F64, oprec, &transa, &transb,
        &m, &n, &k, &dalpha, a, &lda, b, &ldb, &dbeta, c, &ldc);
    } break;
    case LIBXSMM_GEMM_PRECISION_F32: {
      const float alpha = (float)dalpha, beta = (float)dbeta;
      libxsmm_gemm_print2(ostream, LIBXSMM_GEMM_PRECISION_F32, oprec, &transa, &transb,
        &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    } break;
    default: {
      libxsmm_gemm_print2(ostream, iprec, oprec, &transa, &transb,
        &m, &n, &k, &dalpha, a, &lda, b, &ldb, &dbeta, c, &ldc);
    }
  }
}


LIBXSMM_API void libxsmm_gemm_xprint(void* ostream,
  libxsmm_xmmfunction kernel, const void* a, const void* b, void* c)
{
  libxsmm_mmkernel_info info;
  size_t code_size;
  if (EXIT_SUCCESS == libxsmm_get_mmkernel_info(kernel, &info, &code_size)) {
    libxsmm_code_pointer code_pointer;
    libxsmm_gemm_dprint2(ostream, info.iprecision, info.oprecision,
      (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & info.flags) ? 'N' : 'T'),
      (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & info.flags) ? 'N' : 'T'), (libxsmm_blasint)info.m, (libxsmm_blasint)info.n, (libxsmm_blasint)info.k,
      /*0 != (LIBXSMM_GEMM_FLAG_ALPHA_0 & libxsmm_gemm_batchdesc.flags) ? 0 : */1, a, (libxsmm_blasint)info.lda, b, (libxsmm_blasint)info.ldb,
      0 != (LIBXSMM_GEMM_FLAG_BETA_0  & libxsmm_gemm_batchdesc.flags) ? 0 : 1, c, (libxsmm_blasint)info.ldc);
    code_pointer.xgemm = kernel; fprintf((FILE*)ostream, " = %p+%u", code_pointer.ptr_const, (unsigned int)code_size);
  }
}


LIBXSMM_API void libxsmm_blas_xgemm(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_INIT
  switch (iprec) {
    case LIBXSMM_GEMM_PRECISION_F64: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_BLAS_XGEMM(double, double, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    case LIBXSMM_GEMM_PRECISION_F32: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_BLAS_XGEMM(float, float, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    default: if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      static int error_once = 0;
      LIBXSMM_UNUSED(oprec);
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) { /* TODO: support I16, etc. */
        fprintf(stderr, "LIBXSMM ERROR: unsupported data-type requested!\n");
      }
    }
  }
}


LIBXSMM_API_INLINE int libxsmm_gemm_plan_internal(unsigned int nthreads,
  unsigned int m, unsigned int n, unsigned int k, unsigned int tm, unsigned int tn, unsigned int tk,
  unsigned int* nmt, unsigned int* nnt, unsigned int* nkt,
  unsigned int* mt, unsigned int* nt, unsigned int* kt)
{
  unsigned int result = EXIT_SUCCESS, replan = 0;
  LIBXSMM_ASSERT(NULL != nmt && NULL != nnt && NULL != nkt);
  LIBXSMM_ASSERT(NULL != mt && NULL != nt && NULL != kt);
  LIBXSMM_ASSERT(0 < nthreads);
  *nmt = (m + tm - 1) / tm;
  *nnt = (n + tn - 1) / tn;
  *nkt = (k + tk - 1) / tk;
  do {
    if (1 >= replan) *mt = libxsmm_product_limit(*nmt, nthreads, 0);
    if (1 == replan || nthreads <= *mt) { /* M-parallelism */
      *nt = 1;
      *kt = 1;
      replan = 0;
    }
    else {
      const unsigned int mntasks = libxsmm_product_limit((*nmt) * (*nnt), nthreads, 0);
      if (0 == replan && *mt >= mntasks) replan = 1;
      if (2 == replan || (0 == replan && nthreads <= mntasks)) { /* MN-parallelism */
        *nt = mntasks / *mt;
        *kt = 1;
        replan = 0;
      }
      else { /* MNK-parallelism */
        const unsigned int mnktasks = libxsmm_product_limit((*nmt) * (*nnt) * (*nkt), nthreads, 0);
        if (mntasks < mnktasks) {
#if defined(LIBXSMM_GEMM_KPARALLEL)
          *nt = mntasks / *mt;
          *kt = mnktasks / mntasks;
          replan = 0;
#else
          static int error_once = 0;
          if ((1 < libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
            && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXSMM WARNING: K-parallelism triggered!\n");
          }
#endif
        }
#if defined(LIBXSMM_GEMM_KPARALLEL)
        else
#endif
        if (0 == replan) replan = 2;
      }
    }
  } while (0 != replan);
  if (0 == *nt || 0 == *nt || 0 == *kt) {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API libxsmm_gemm_handle* libxsmm_gemm_handle_init(libxsmm_gemm_blob* blob, size_t* scratch_size,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta, int flags, /*unsigned*/int nthreads)
{
  unsigned int ulda, uldb, um, un, uk, tm = 0, tn = 0, tk = 0, ntasks = 0;
  libxsmm_descriptor_blob desc_blob;
  union {
    libxsmm_gemm_handle* ptr;
    libxsmm_gemm_blob* blob;
  } result;
  LIBXSMM_ASSERT(sizeof(libxsmm_gemm_handle) <= sizeof(libxsmm_gemm_blob));
  if (NULL != blob && NULL != scratch_size && NULL != m && 0 < nthreads) {
    unsigned int ntm = 0, ntn = 0, ntk = 0, mt = 0, nt = 0, kt = 0;
    const char *const env_tm = getenv("LIBXSMM_TGEMM_M");
    libxsmm_blasint klda, kldb, kldc, km, kn;
    libxsmm_gemm_descriptor* desc;
    const int prf_copy = 0;
    double dbeta;
    LIBXSMM_INIT
    result.blob = blob;
#if defined(NDEBUG)
    result.ptr->copy_a.pmm = result.ptr->copy_b.pmm = result.ptr->copy_i.pmm = result.ptr->copy_o.pmm = NULL;
#else
    memset(blob, 0, sizeof(libxsmm_gemm_blob));
#endif
    if (EXIT_SUCCESS != libxsmm_gemm_dvalue(oprec, beta, &dbeta) && LIBXSMM_NEQ(0, dbeta)) dbeta = LIBXSMM_BETA; /* fuse beta into flags */
    result.ptr->gemm_flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS) | (LIBXSMM_NEQ(0, dbeta) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0);
    /* TODO: check that arguments fit into handle (unsigned int vs. libxsmm_blasint) */
    uk = (unsigned int)(NULL != k ? *k : *m); un = (NULL != n ? ((unsigned int)(*n)) : uk); um = (unsigned int)(*m);
    LIBXSMM_ASSERT(LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags) || um == un);
    result.ptr->otypesize = libxsmm_gemm_typesize(oprec);
    result.ptr->nthreads = (unsigned int)nthreads;
    if (NULL == env_tm || 0 >= atoi(env_tm)) {
      const unsigned int vwidth = LIBXSMM_MAX(internal_gemm_vwidth / result.ptr->otypesize, 1);
      unsigned int tmi = libxsmm_product_limit(um, LIBXSMM_GEMM_MSIZE_MAX, 0);
      for (; vwidth <= tmi; tmi = libxsmm_product_limit(um, tmi - 1, 0)) {
        unsigned int tni = libxsmm_product_limit(un, LIBXSMM_MAX((unsigned int)(tmi * internal_gemm_nstretch), 1), 1);
        unsigned int tki = libxsmm_product_limit(uk, LIBXSMM_MAX((unsigned int)(tmi * internal_gemm_kstretch), 1), 1);
        unsigned int ntmi, ntni, ntki, mti, nti, kti;
        LIBXSMM_ASSERT(tmi <= um && tni <= un && tki <= uk);
        if (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags)) {
          const unsigned int ttm = (unsigned int)libxsmm_product_limit(tmi, result.ptr->nthreads, 0);
          const unsigned int ttn = (unsigned int)libxsmm_product_limit(tni, result.ptr->nthreads, 0);
          tmi = tni = LIBXSMM_MIN(ttm, ttn); /* prefer threads over larger tile */
        }
        if (EXIT_SUCCESS == libxsmm_gemm_plan_internal(result.ptr->nthreads, um, un, uk, tmi, tni, tki,
          &ntmi, &ntni, &ntki, &mti, &nti, &kti))
        {
          const int exit_plan = ((tmi < um && tni < un && tki < uk && (tm != tmi || tn != tni || tk != tki)) ? 0 : 1);
          const unsigned itasks = mti * nti * kti;
          LIBXSMM_ASSERT(1 <= itasks);
          if (ntasks < itasks) {
            ntm = ntmi; ntn = ntni; ntk = ntki;
            mt = mti; nt = nti; kt = kti;
            tm = tmi; tn = tni; tk = tki;
            ntasks = itasks;
          }
          if (result.ptr->nthreads == itasks || 0 != exit_plan) break;
        }
      }
    }
    else {
      const unsigned int tmi = atoi(env_tm);
      tm = libxsmm_product_limit(um, LIBXSMM_MIN(tmi, LIBXSMM_GEMM_MSIZE_MAX), 0);
      tn = libxsmm_product_limit(un, LIBXSMM_MAX((unsigned int)(tm * internal_gemm_nstretch), 1), 1);
      tk = libxsmm_product_limit(uk, LIBXSMM_MAX((unsigned int)(tm * internal_gemm_kstretch), 1), 1);
      if (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags)) {
        const unsigned int ttm = (unsigned int)libxsmm_product_limit(tm, result.ptr->nthreads, 0);
        const unsigned int ttn = (unsigned int)libxsmm_product_limit(tn, result.ptr->nthreads, 0);
        tm = tn = LIBXSMM_MIN(ttm, ttn); /* prefer threads over larger tile */
      }
      if (EXIT_SUCCESS == libxsmm_gemm_plan_internal(result.ptr->nthreads, um, un, uk, tm, tn, tk,
        &ntm, &ntn, &ntk, &mt, &nt, &kt))
      {
        ntasks = mt * nt * kt;
      }
    }
    LIBXSMM_ASSERT(LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags) || tm == tn);
    if (0 == ntasks || 0 != (um % tm) || 0 != (un % tn) || 0 != (uk % tk)) {
      return NULL;
    }
    result.ptr->flags = flags;
    if (LIBXSMM_GEMM_HANDLE_FLAG_COPY_AUTO == flags && /* check for big enough problem size */
      ((unsigned long long)(LIBXSMM_MAX_MNK)) < LIBXSMM_MNK_SIZE(um, un, uk))
    {
      result.ptr->flags |= LIBXSMM_GEMM_HANDLE_FLAG_COPY_C;
    }
    result.ptr->itypesize = libxsmm_gemm_typesize(iprec);
    result.ptr->ldc = (unsigned int)(NULL != ldc ? *ldc : *m);
    ulda = (NULL != lda ? ((unsigned int)(*lda)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & result.ptr->gemm_flags) ? ((unsigned int)(*m)) : uk));
    uldb = (NULL != ldb ? ((unsigned int)(*ldb)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & result.ptr->gemm_flags) ? uk : un));
    if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags)) { /* NN, NT, or TN */
      kldc = (libxsmm_blasint)result.ptr->ldc;
      klda = (libxsmm_blasint)ulda;
      kldb = (libxsmm_blasint)uldb;
      if (0 != (LIBXSMM_GEMM_FLAG_TRANS_A & result.ptr->gemm_flags)) { /* TN */
#if !defined(LIBXSMM_GEMM_NOJIT_TRANS)
        result.ptr->copy_a.xtrans = libxsmm_dispatch_trans(libxsmm_trans_descriptor_init(&desc_blob,
          result.ptr->itypesize, tk, tm, tm/*ldo*/));
#endif
        klda = (libxsmm_blasint)tm;
      }
      else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_A & result.ptr->flags)) {
        result.ptr->copy_a.xmatcopy = libxsmm_dispatch_mcopy(libxsmm_mcopy_descriptor_init(&desc_blob,
          result.ptr->itypesize, tm, tk, tm/*ldo*/, ulda/*ldi*/,
          0/*flags*/, prf_copy, NULL/*unroll*/));
        klda = (libxsmm_blasint)tm;
      }
      if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & result.ptr->gemm_flags)) { /* NT */
#if !defined(LIBXSMM_GEMM_NOJIT_TRANS)
        result.ptr->copy_b.xtrans = libxsmm_dispatch_trans(libxsmm_trans_descriptor_init(&desc_blob,
          result.ptr->itypesize, tn, tk, tk/*ldo*/));
#endif
        kldb = (libxsmm_blasint)tk;
      }
      else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_B & result.ptr->flags)) {
        result.ptr->copy_b.xmatcopy = libxsmm_dispatch_mcopy(libxsmm_mcopy_descriptor_init(&desc_blob,
          result.ptr->itypesize, tk, tn, tk/*ldo*/, uldb/*ldi*/,
          0/*flags*/, prf_copy, NULL/*unroll*/));
        kldb = (libxsmm_blasint)tk;
      }
      if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_C & result.ptr->flags)) {
        result.ptr->copy_o.xmatcopy = libxsmm_dispatch_mcopy(libxsmm_mcopy_descriptor_init(&desc_blob,
          result.ptr->otypesize, tm, tn, result.ptr->ldc/*ldo*/, tm/*ldi*/,
          0/*flags*/, prf_copy, NULL/*unroll*/));
        if (0 == (result.ptr->gemm_flags & LIBXSMM_GEMM_FLAG_BETA_0)) { /* copy-in only if beta!=0 */
          result.ptr->copy_i.xmatcopy = libxsmm_dispatch_mcopy(libxsmm_mcopy_descriptor_init(&desc_blob,
            result.ptr->otypesize, tm, tn, tm/*ldo*/, result.ptr->ldc/*ldi*/,
            0/*flags*/, prf_copy, NULL/*unroll*/));
        }
        kldc = (libxsmm_blasint)tm;
      }
      result.ptr->lda = ulda; result.ptr->ldb = uldb;
      result.ptr->tm = tm; result.ptr->tn = tn;
      result.ptr->mt = mt; result.ptr->nt = nt;
      result.ptr->m = um; result.ptr->n = un;
      result.ptr->dm = ntm / mt * tm;
      result.ptr->dn = ntn / nt * tn;
      km = tm; kn = tn;
    }
    else { /* TT */
      const unsigned int tt = tm;
      klda = (libxsmm_blasint)uldb;
      kldb = (libxsmm_blasint)ulda;
      kldc = (libxsmm_blasint)tt;
      LIBXSMM_ASSERT(tt == tn);
#if !defined(LIBXSMM_GEMM_NOJIT_TRANS)
      result.ptr->copy_o.xtrans = libxsmm_dispatch_trans(libxsmm_trans_descriptor_init(&desc_blob,
        result.ptr->otypesize, tt, tt, result.ptr->ldc/*ldo*/));
      if (0 == (result.ptr->gemm_flags & LIBXSMM_GEMM_FLAG_BETA_0)) { /* copy-in only if beta!=0 */
        result.ptr->copy_i.xtrans = libxsmm_dispatch_trans(libxsmm_trans_descriptor_init(&desc_blob,
          result.ptr->otypesize, tt, tt, tt/*ldo*/));
      }
#endif
      if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_A & result.ptr->flags)) {
        result.ptr->copy_a.xmatcopy = libxsmm_dispatch_mcopy(libxsmm_mcopy_descriptor_init(&desc_blob,
          result.ptr->itypesize, tt, tk, tk/*ldo*/, uldb/*ldi*/,
          0/*flags*/, prf_copy, NULL/*unroll*/));
        klda = (libxsmm_blasint)tt;
      }
      if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_B & result.ptr->flags)) {
        result.ptr->copy_b.xmatcopy = libxsmm_dispatch_mcopy(libxsmm_mcopy_descriptor_init(&desc_blob,
          result.ptr->itypesize, tk, tn, tk/*ldo*/, ulda/*ldi*/,
          0/*flags*/, prf_copy, NULL/*unroll*/));
        kldb = (libxsmm_blasint)tk;
      }
      result.ptr->lda = uldb; result.ptr->ldb = ulda;
      result.ptr->tm = tn; result.ptr->tn = tm;
      result.ptr->mt = nt; result.ptr->nt = mt;
      result.ptr->m = un; result.ptr->n = um;
      result.ptr->dm = ntn / nt * tn;
      result.ptr->dn = ntm / mt * tm;
      km = kn = tt;
    }
    result.ptr->dk = ntk / kt * tk;
    result.ptr->tk = tk;
    result.ptr->kt = kt;
    result.ptr->k = uk;
    desc = libxsmm_gemm_descriptor_init2( /* remove transpose flags from kernel request */
      &desc_blob, iprec, oprec, km, kn, result.ptr->tk, klda, kldb, kldc,
      alpha, beta, result.ptr->gemm_flags & ~LIBXSMM_GEMM_FLAG_TRANS_AB, internal_gemm_tiled_prefetch);
    result.ptr->kernel[0] = libxsmm_xmmdispatch(desc);
    if (NULL != result.ptr->kernel[0].xmm) {
      if (0 == (desc->flags & LIBXSMM_GEMM_FLAG_BETA_0)) { /* beta!=0 */
        result.ptr->kernel[1] = result.ptr->kernel[0];
      }
      else { /* generate kernel with beta=1 */
        desc->flags &= ~LIBXSMM_GEMM_FLAG_BETA_0;
        result.ptr->kernel[1] = libxsmm_xmmdispatch(desc);
        if (NULL == result.ptr->kernel[1].xmm) result.ptr = NULL;
      }
    }
    else {
      result.ptr = NULL;
    }
    if (NULL != result.ptr) { /* thread-local scratch buffer for GEMM */
      result.ptr->size_a = result.ptr->size_b = result.ptr->size_c = 0;
      if (0 != (result.ptr->flags & LIBXSMM_GEMM_HANDLE_FLAG_COPY_A)
        || (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags) &&
           (LIBXSMM_GEMM_FLAG_TRANS_A & result.ptr->gemm_flags) != 0))
      {
        const size_t size_a = result.ptr->tm * result.ptr->tk * result.ptr->itypesize;
        result.ptr->size_a = LIBXSMM_UP2(size_a, LIBXSMM_CACHELINE);
      }
      if (0 != (result.ptr->flags & LIBXSMM_GEMM_HANDLE_FLAG_COPY_B)
        || (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags) &&
           (LIBXSMM_GEMM_FLAG_TRANS_B & result.ptr->gemm_flags) != 0))
      {
        const size_t size_b = result.ptr->tk * result.ptr->tn * result.ptr->itypesize;
        result.ptr->size_b = LIBXSMM_UP2(size_b, LIBXSMM_CACHELINE);
      }
      if (0 != (result.ptr->flags & LIBXSMM_GEMM_HANDLE_FLAG_COPY_C)
        || LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags))
      {
        const size_t size_c = result.ptr->tm * result.ptr->tn * result.ptr->otypesize;
        result.ptr->size_c = LIBXSMM_UP2(size_c, LIBXSMM_CACHELINE);
      }
      *scratch_size = ntasks * (result.ptr->size_a + result.ptr->size_b + result.ptr->size_c);
    }
    else {
      *scratch_size = 0;
    }
  }
  else {
    result.ptr = NULL;
  }
  return result.ptr;
}


LIBXSMM_API void libxsmm_gemm_thread(const libxsmm_gemm_handle* handle, void* scratch,
  const void* a, const void* b, void* c, /*unsigned*/int tid)
{
#if !defined(NDEBUG)
  if (NULL != handle && 0 <= tid)
#endif
  {
    const unsigned int ntasks = handle->mt * handle->nt * handle->kt;
    const unsigned int spread = handle->nthreads / ntasks;
    const unsigned int utid = (unsigned int)tid, vtid = utid / spread;
    if (utid < (spread * ntasks) && 0 == (utid - vtid * spread)) {
      const unsigned int rtid = vtid / handle->mt, mtid = vtid - rtid * handle->mt, ntid = rtid % handle->nt, ktid = vtid / (handle->mt * handle->nt);
      const unsigned int m0 = mtid * handle->dm, m1 = LIBXSMM_MIN(m0 + handle->dm, handle->m);
      const unsigned int n0 = ntid * handle->dn, n1 = LIBXSMM_MIN(n0 + handle->dn, handle->n);
      const unsigned int k0 = ktid * handle->dk, k1 = LIBXSMM_MIN(k0 + handle->dk, handle->k);
      const unsigned int ldo = (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags) ? handle->tm : handle->tk);
      /* calculate increments to simplify address calculations */
      const unsigned int dom = handle->tm * handle->otypesize;
      const unsigned int don = handle->tn * handle->otypesize;
      const unsigned int dik = handle->tk * handle->itypesize;
      const unsigned int on = handle->otypesize * n0;
      /* calculate base address of thread-local storage */
      char *const at = (char*)scratch + (handle->size_a + handle->size_b + handle->size_c) * vtid;
      char *const bt = at + handle->size_a;
      char *const ct = bt + handle->size_b;
      /* loop induction variables and other variables */
      unsigned int om = handle->otypesize * m0, im = m0, in = n0, ik = k0, im1, in1, ik1;
      LIBXSMM_ASSERT_MSG(mtid < handle->mt && ntid < handle->nt && ktid < handle->kt, "Invalid task ID!");
      LIBXSMM_ASSERT_MSG(m1 <= handle->m && n1 <= handle->n && k1 <= handle->k, "Invalid task size!");
      for (im1 = im + handle->tm; (im1 - 1) < m1; im = im1, im1 += handle->tm, om += dom) {
        unsigned int dn = don, dka = dik, dkb = dik;
        char *c0 = (char*)c, *ci;
        const char *aa;
        if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags)) {
          if (0 != (LIBXSMM_GEMM_FLAG_TRANS_A & handle->gemm_flags)) { /* TN */
            aa = (const char*)a + ((size_t)im * handle->lda + k0) * handle->itypesize;
          }
          else if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & handle->gemm_flags)) { /* NT */
            aa = (const char*)a + ((size_t)k0 * handle->lda + im) * handle->itypesize;
            dka *= handle->lda; dkb *= handle->ldb;
          }
          else { /* NN */
            aa = (const char*)a + ((size_t)k0 * handle->lda + im) * handle->itypesize;
            dka *= handle->lda;
          }
          c0 += (size_t)on * handle->ldc + om;
          dn *= handle->ldc;
        }
        else { /* TT */
          aa = (const char*)b + ((size_t)k0 * handle->lda + im) * handle->itypesize;
          c0 += (size_t)on + handle->ldc * (size_t)om;
          dka *= handle->lda;
        }
        for (in = n0, in1 = in + handle->tn; (in1 - 1) < n1; in = in1, in1 += handle->tn, c0 += dn) {
          const char *a0 = aa, *b0 = (const char*)b;
          if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags)) {
            if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & handle->gemm_flags)) { /* NT */
              b0 += ((size_t)k0 * handle->ldb + in) * handle->itypesize;
            }
            else { /* NN or TN */
              b0 += ((size_t)in * handle->ldb + k0) * handle->itypesize;
            }
          }
          else { /* TT */
            b0 = (const char*)a + ((size_t)in * handle->ldb + k0) * handle->itypesize;
          }
          if (NULL == handle->copy_i.ptr_const) {
            ci = (NULL == handle->copy_o.ptr_const ? c0 : ct);
            if (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags)) {
              const unsigned int km = handle->tn, kn = handle->tm;
              libxsmm_otrans_internal(ct/*out*/, c0/*in*/, handle->otypesize, handle->ldc/*ldi*/, kn/*ldo*/,
                0, km, 0, kn, km/*tile*/, kn/*tile*/, NULL/*kernel*/);
              ci = ct;
            }
            else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_C & handle->flags)) {
              if (0 == (handle->gemm_flags & LIBXSMM_GEMM_FLAG_BETA_0)) { /* copy-in only if beta!=0 */
                libxsmm_matcopy_internal(ct/*out*/, c0/*in*/, handle->otypesize, handle->ldc/*ldi*/, handle->tm/*ldo*/,
                  0, handle->tm, 0, handle->tn, handle->tm/*tile*/, handle->tn/*tile*/, NULL/*kernel*/);
              }
              ci = ct;
            }
          }
          else { /* MCOPY/TCOPY kernel */
            handle->copy_i.xmatcopy(c0, &handle->ldc, ct, &handle->tm);
            ci = ct;
          }
          for (ik = k0, ik1 = ik + handle->tk; (ik1 - 1) < k1; ik = ik1, ik1 += handle->tk) {
            const char *const a1 = a0 + dka, *const b1 = b0 + dkb, *ai = a0, *bi = b0;
            if (NULL == handle->copy_a.ptr_const) {
              if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags) &&
                 (LIBXSMM_GEMM_FLAG_TRANS_A & handle->gemm_flags) != 0) /* pure A-transpose */
              {
                LIBXSMM_ASSERT(ldo == handle->tm);
                libxsmm_otrans_internal(at/*out*/, a0/*in*/, handle->itypesize, handle->lda/*ldi*/, ldo,
                  0, handle->tk, 0, handle->tm, handle->tk/*tile*/, handle->tm/*tile*/, NULL/*kernel*/);
                ai = at;
              }
              else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_A & handle->flags)) {
                libxsmm_matcopy_internal(at/*out*/, a0/*in*/, handle->itypesize, handle->lda/*ldi*/, ldo,
                  0, handle->tm, 0, handle->tk, handle->tm/*tile*/, handle->tk/*tile*/, NULL/*kernel*/);
                ai = at;
              }
            }
            else { /* MCOPY/TCOPY kernel */
              handle->copy_a.xmatcopy(a0, &handle->lda, at, &ldo);
              ai = at;
            }
            if (NULL == handle->copy_b.ptr_const) {
              if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags) &&
                 (LIBXSMM_GEMM_FLAG_TRANS_B & handle->gemm_flags) != 0) /* pure B-transpose */
              {
                libxsmm_otrans_internal(bt/*out*/, b0/*in*/, handle->itypesize, handle->ldb/*ldi*/, handle->tk/*ldo*/,
                  0, handle->tn, 0, handle->tk, handle->tn/*tile*/, handle->tk/*tile*/, NULL/*kernel*/);
                bi = bt;
              }
              else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_B & handle->flags)) {
                libxsmm_matcopy_internal(bt/*out*/, b0/*in*/, handle->itypesize, handle->ldb/*ldi*/, handle->tk/*ldo*/,
                  0, handle->tk, 0, handle->tn, handle->tk/*tile*/, handle->tn/*tile*/, NULL/*kernel*/);
                bi = bt;
              }
            }
            else { /* MCOPY/TCOPY kernel */
              handle->copy_b.xmatcopy(b0, &handle->ldb, bt, &handle->tk);
              bi = bt;
            }
            /* beta0-kernel on first-touch, beta1-kernel otherwise (beta0/beta1 are identical if beta=1) */
            LIBXSMM_MMCALL_PRF(handle->kernel[k0!=ik?1:0].xmm, ai, bi, ci, a1, b1, c0);
            a0 = a1;
            b0 = b1;
          }
          /* TODO: synchronize */
          if (NULL == handle->copy_o.ptr_const) {
            if (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags)) {
              libxsmm_otrans_internal(c0/*out*/, ct/*in*/, handle->otypesize, handle->tm/*ldi*/, handle->ldc/*ldo*/,
                0, handle->tm, 0, handle->tn, handle->tm/*tile*/, handle->tn/*tile*/, NULL/*kernel*/);
            }
            else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_C & handle->flags)) {
              libxsmm_matcopy_internal(c0/*out*/, ct/*in*/, handle->otypesize, handle->tm/*ldi*/, handle->ldc/*ldo*/,
                0, handle->tm, 0, handle->tn, handle->tm/*tile*/, handle->tn/*tile*/, NULL/*kernel*/);
            }
          }
          else { /* MCOPY/TCOPY kernel */
            handle->copy_o.xmatcopy(ct, &handle->tm, c0, &handle->ldc);
          }
        }
      }
    }
  }
#if !defined(NDEBUG)
  else if (/*implies LIBXSMM_INIT*/0 != libxsmm_get_verbosity()) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_gemm_thread - invalid handle!\n");
    }
  }
#endif
}


LIBXSMM_API void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_XGEMM(double, double, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_XGEMM(float, float, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API void libxsmm_wigemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const int* alpha, const short* a, const libxsmm_blasint* lda,
  const short* b, const libxsmm_blasint* ldb,
  const int* beta, int* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_XGEMM(short, int, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API void libxsmm_wsgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const short* a, const libxsmm_blasint* lda,
  const short* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_XGEMM(short, float, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API int libxsmm_mmbatch_internal(libxsmm_xmmfunction kernel, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize, /*unsigned*/int tid, /*unsigned*/int nthreads,
  const libxsmm_gemm_descriptor* info)
{
  int result = EXIT_SUCCESS;
  const libxsmm_blasint size = LIBXSMM_ABS(batchsize);
  const libxsmm_blasint tasksize = (size + nthreads - 1) / nthreads;
  const libxsmm_blasint begin = tid * tasksize, span = begin + tasksize;
  const libxsmm_blasint end = LIBXSMM_MIN(span, size);

  LIBXSMM_ASSERT(0 != info);
  if (begin < end) {
    const libxsmm_blasint typesize = libxsmm_gemm_typesize((libxsmm_gemm_precision)info->datatype);
    const char *const a0 = (const char*)a, *const b0 = (const char*)b;
    char *const c0 = (char*)c;
    libxsmm_blasint i, ni;

    LIBXSMM_ASSERT(0 < typesize);
    if (0 != index_stride) { /* stride arrays contain indexes */
#if defined(LIBXSMM_GEMM_CHECK)
      if (((int)sizeof(libxsmm_blasint)) <= index_stride)
#endif
      {
        const char *const sa = (const char*)stride_a, *const sb = (const char*)stride_b, *const sc = (const char*)stride_c;
        libxsmm_blasint ii = begin * index_stride, ic = (0 != sc ? (*((const libxsmm_blasint*)(sc + ii)) - index_base) : 0);
        const char* ai = a0 + (0 != sa ? ((*((const libxsmm_blasint*)(sa + ii)) - index_base) * typesize) : 0);
        const char* bi = b0 + (0 != sb ? ((*((const libxsmm_blasint*)(sb + ii)) - index_base) * typesize) : 0);
        char*       ci = c0 + (size_t)ic * typesize;
        const libxsmm_blasint end1 = (end != size ? end : (end - 1));
#if !defined(LIBXSMM_NO_SYNC)
        if (1 == nthreads || 0 == internal_gemm_nlocks || 0 > batchsize || 0 != (LIBXSMM_GEMM_FLAG_BETA_0 & info->flags))
#endif
        { /* no locking */
          if (0 != sa && 0 != sb && 0 != sc) {
            for (i = begin; i < end1; i = ni) {
              ni = i + 1; ii = ni * index_stride;
              {
                const char *const an = a0 + (size_t)(*(const libxsmm_blasint*)(sa + ii) - (size_t)index_base) * typesize;
                const char *const bn = b0 + (size_t)(*(const libxsmm_blasint*)(sb + ii) - (size_t)index_base) * typesize;
                char       *const cn = c0 + (size_t)(*(const libxsmm_blasint*)(sc + ii) - (size_t)index_base) * typesize;
                kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
                ai = an; bi = bn; ci = cn;
              }
            }
          }
          else {
            for (i = begin; i < end1; i = ni) {
              ni = i + 1; ii = ni * index_stride;
              {
                const char *const an = a0 + (0 != sa ? ((size_t)(*(const libxsmm_blasint*)(sa + ii) - (size_t)index_base) * typesize) : 0);
                const char *const bn = b0 + (0 != sb ? ((size_t)(*(const libxsmm_blasint*)(sb + ii) - (size_t)index_base) * typesize) : 0);
                char       *const cn = c0 + (0 != sc ? ((size_t)(*(const libxsmm_blasint*)(sc + ii) - (size_t)index_base) * typesize) : 0);
                kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
                ai = an; bi = bn; ci = cn;
              }
            }
          }
          if (end != end1) { /* remainder multiplication */
            kernel.xmm(ai, bi, ci, ai, bi, ci); /* pseudo-prefetch */
          }
        }
#if !defined(LIBXSMM_NO_SYNC)
        else { /* synchronize among C-indexes */
          LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK)* lock = &internal_gemm_lock[LIBXSMM_GEMM_LOCKIDX(ic, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
          LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK)* lock0 = 0;
# endif
          LIBXSMM_ASSERT(0 != lock);
          if (0 != sa && 0 != sb && 0 != sc) {
            for (i = begin; i < end1; i = ni) {
              ni = i + 1; ii = ni * index_stride; ic = (*((const libxsmm_blasint*)(sc + ii)) - index_base);
              {
                const char *const an = a0 + (size_t)(*((const libxsmm_blasint*)(sa + ii)) - (size_t)index_base) * typesize;
                const char *const bn = b0 + (size_t)(*((const libxsmm_blasint*)(sb + ii)) - (size_t)index_base) * typesize;
                char       *const cn = c0 + (size_t)ic * typesize;
                LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) *const lock1 = &internal_gemm_lock[LIBXSMM_GEMM_LOCKIDX(ic, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
                if (lock != lock0) { lock0 = lock; LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock); }
# else
                LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
# endif
                kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
# if defined(LIBXSMM_GEMM_LOCKFWD)
                if (lock != lock1 || ni == end1) { LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1; }
# else
                LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1;
# endif
                ai = an; bi = bn; ci = cn; /* next */
              }
            }
          }
          else {
            for (i = begin; i < end1; i = ni) {
              ni = i + 1; ii = ni * index_stride; ic = (0 != sc ? (*((const libxsmm_blasint*)(sc + ii)) - index_base) : 0);
              {
                const char *const an = a0 + (0 != sa ? ((*((const libxsmm_blasint*)(sa + ii)) - index_base) * typesize) : 0);
                const char *const bn = b0 + (0 != sb ? ((*((const libxsmm_blasint*)(sb + ii)) - index_base) * typesize) : 0);
                char       *const cn = c0 + (size_t)ic * typesize;
                LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) *const lock1 = &internal_gemm_lock[LIBXSMM_GEMM_LOCKIDX(ic, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
                if (lock != lock0) { lock0 = lock; LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock); }
# else
                LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
# endif
                kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
# if defined(LIBXSMM_GEMM_LOCKFWD)
                if (lock != lock1 || ni == end1) { LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1; }
# else
                LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1;
# endif
                ai = an; bi = bn; ci = cn; /* next */
              }
            }
          }
          if (end != end1) { /* remainder multiplication */
            LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
            kernel.xmm(ai, bi, ci, ai, bi, ci); /* pseudo-prefetch */
            LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock);
          }
        }
#endif /*!defined(LIBXSMM_NO_SYNC)*/
      }
#if defined(LIBXSMM_GEMM_CHECK)
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
#endif
    }
    else { /* singular strides are measured in Bytes */
      const libxsmm_blasint stride_unit = sizeof(libxsmm_blasint);
      const libxsmm_blasint da = (0 != stride_a ? (*stride_a - index_base * stride_unit) : 0);
      const libxsmm_blasint db = (0 != stride_b ? (*stride_b - index_base * stride_unit) : 0);
      const libxsmm_blasint dc = (0 != stride_c ? (*stride_c - index_base * stride_unit) : 0);
#if defined(LIBXSMM_GEMM_CHECK)
      if (typesize <= da && typesize <= db && typesize <= dc)
#endif
      {
        const libxsmm_blasint end1 = (end != size ? end : (end - 1));
        const char *ai = a0 + (size_t)da * begin, *bi = b0 + (size_t)db * begin;
        char* ci = c0 + (size_t)dc * begin;

#if !defined(LIBXSMM_NO_SYNC)
        if (1 == nthreads || 0 == internal_gemm_nlocks || 0 > batchsize || 0 != (LIBXSMM_GEMM_FLAG_BETA_0 & info->flags))
#endif
        { /* no locking */
          for (i = begin; i < end1; ++i) {
#if defined(LIBXSMM_GEMM_CHECK)
            if (0 != *((const void**)ai) && 0 != *((const void**)bi) && 0 != *((const void**)ci))
#endif
            {
              const char *const an = ai + da, *const bn = bi + db;
              char *const cn = ci + dc;
              kernel.xmm( /* with prefetch */
                *((const void**)ai), *((const void**)bi), *((void**)ci),
                *((const void**)an), *((const void**)bn), *((const void**)cn));
              ai = an; bi = bn; ci = cn; /* next */
            }
          }
          if (end != end1 /* remainder multiplication */
#if defined(LIBXSMM_GEMM_CHECK)
            && 0 != *((const void**)ai) && 0 != *((const void**)bi) && 0 != *((const void**)ci)
#endif
            )
          {
            kernel.xmm( /* pseudo-prefetch */
              *((const void**)ai), *((const void**)bi), *((void**)ci),
              *((const void**)ai), *((const void**)bi), *((const void**)ci));
          }
        }
#if !defined(LIBXSMM_NO_SYNC)
        else { /* synchronize among C-indexes */
          void* cc = *((void**)ci);
          LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK)* lock = &internal_gemm_lock[LIBXSMM_GEMM_LOCKPTR(cc, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
          LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK)* lock0 = 0;
# endif
          LIBXSMM_ASSERT(0 != lock);
          for (i = begin; i < end1; i = ni) {
            ni = i + 1;
# if defined(LIBXSMM_GEMM_CHECK)
            if (0 != *((const void**)ai) && 0 != *((const void**)bi) && 0 != cc)
# endif
            {
              const char *const an = ai + da, *const bn = bi + db;
              char *const cn = ci + dc;
              void *const nc = *((void**)cn);
              LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) *const lock1 = &internal_gemm_lock[LIBXSMM_GEMM_LOCKPTR(nc, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
              if (lock != lock0) { lock0 = lock; LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock); }
# else
              LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
# endif
              kernel.xmm( /* with prefetch */
                *((const void**)ai), *((const void**)bi), cc,
                *((const void**)an), *((const void**)bn), *((const void**)cn));
# if defined(LIBXSMM_GEMM_LOCKFWD)
              if (lock != lock1 || ni == end1) { LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1; }
# else
              LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1;
# endif
              ai = an; bi = bn; ci = cn; cc = nc; /* next */
            }
          }
          if (end != end1 /* remainder multiplication */
# if defined(LIBXSMM_GEMM_CHECK)
            && 0 != *((const void**)ai) && 0 != *((const void**)bi) && 0 != cc
# endif
            )
          {
            LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
            kernel.xmm( /* pseudo-prefetch */
              *((const void**)ai), *((const void**)bi), cc,
              *((const void**)ai), *((const void**)bi), cc);
            LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock);
          }
        }
#endif /*!defined(LIBXSMM_NO_SYNC)*/
      }
#if defined(LIBXSMM_GEMM_CHECK)
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
#endif
    }
  }
  return result;
}


LIBXSMM_API int libxsmm_mmbatch(libxsmm_xmmfunction kernel, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize, /*unsigned*/int tid, /*unsigned*/int nthreads)
{
  const libxsmm_kernel_info* info;
  libxsmm_code_pointer code;
  libxsmm_kernel_kind kind;
  int result;
  code.xgemm = kernel;
  info = libxsmm_get_kernel_info(code, &kind, NULL/*size*/);
  if (0 != info && LIBXSMM_KERNEL_KIND_MATMUL == kind && 0 != a && 0 != b && 0 != c
    /* use (signed) integer types, but check sanity of input */
    && 0 <= tid && tid < nthreads)
  {
    LIBXSMM_INIT
    result = libxsmm_mmbatch_internal(kernel, index_base, index_stride,
      stride_a, stride_b, stride_c, a, b, c, batchsize,
      tid, nthreads, &info->xgemm);
  }
  else { /* incorrect argument(s) */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_dmmbatch_blas(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const double* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  int result = EXIT_SUCCESS;

  if (0 != a && 0 != b && 0 != c) {
    const libxsmm_blasint end = LIBXSMM_ABS(batchsize);
    libxsmm_blasint i;

    if (0 != index_stride) { /* stride arrays contain indexes */
      if (((int)sizeof(libxsmm_blasint)) <= index_stride) {
        const libxsmm_blasint da = (0 != stride_a ? (*stride_a - index_base) : 0);
        const libxsmm_blasint db = (0 != stride_b ? (*stride_b - index_base) : 0);
        const libxsmm_blasint dc = (0 != stride_c ? (*stride_c - index_base) : 0);
        const char *const sa = (const char*)stride_a, *const sb = (const char*)stride_b, *const sc = (const char*)stride_c;
        const double *const a0 = (const double*)a, *const b0 = (const double*)b, *ai = a0 + da, *bi = b0 + db;
        double *const c0 = (double*)c, *ci = c0 + dc;
        for (i = 0; i < end; ++i) {
          const libxsmm_blasint ii = (i + 1) * index_stride;
          const double *const an = a0 + (0 != stride_a ? (*((const libxsmm_blasint*)(sa + ii)) - index_base) : 0);
          const double *const bn = b0 + (0 != stride_b ? (*((const libxsmm_blasint*)(sb + ii)) - index_base) : 0);
          double       *const cn = c0 + (0 != stride_c ? (*((const libxsmm_blasint*)(sc + ii)) - index_base) : 0);
#if defined(LIBXSMM_GEMM_CHECK)
          if (0 != ai && 0 != bi && 0 != ci)
#endif
          {
            libxsmm_blas_dgemm(transa, transb, &m, &n, &k, alpha, ai, lda, bi, ldb, beta, ci, ldc);
          }
          ai = an; bi = bn; ci = cn;
        }
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
    else { /* singular strides are measured in Bytes */
      const libxsmm_blasint stride_unit = sizeof(libxsmm_blasint);
      const libxsmm_blasint da = (0 != stride_a ? (*stride_a - index_base * stride_unit) : 0);
      const libxsmm_blasint db = (0 != stride_b ? (*stride_b - index_base * stride_unit) : 0);
      const libxsmm_blasint dc = (0 != stride_c ? (*stride_c - index_base * stride_unit) : 0);
      if (((int)sizeof(double)) <= LIBXSMM_MIN(LIBXSMM_MIN(da, db), dc)) {
        const char *const a0 = (const char*)a, *const b0 = (const char*)b, *ai = a0, *bi = b0;
        char *const c0 = (char*)c, *ci = c0;
        for (i = 0; i < end; ++i) {
          const char *const an = ai + da, *const bn = bi + db;
          char *const cn = ci + dc;
#if defined(LIBXSMM_GEMM_CHECK)
          if (0 != *((const double**)ai) && 0 != *((const double**)bi) && 0 != *((const double**)ci))
#endif
          {
            libxsmm_blas_dgemm(transa, transb, &m, &n, &k, alpha, *((const double**)ai), lda, *((const double**)bi), ldb, beta, *((double**)ci), ldc);
          }
          ai = an; bi = bn; ci = cn; /* next */
        }
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
  }
  else { /* incorrect argument(s) */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_smmbatch_blas(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const float* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  int result = EXIT_SUCCESS;

  if (0 != a && 0 != b && 0 != c) {
    const libxsmm_blasint end = LIBXSMM_ABS(batchsize);
    libxsmm_blasint i;

    if (0 != index_stride) { /* stride arrays contain indexes */
      if (((int)sizeof(libxsmm_blasint)) <= index_stride) {
        const libxsmm_blasint da = (0 != stride_a ? (*stride_a - index_base) : 0);
        const libxsmm_blasint db = (0 != stride_b ? (*stride_b - index_base) : 0);
        const libxsmm_blasint dc = (0 != stride_c ? (*stride_c - index_base) : 0);
        const char *const sa = (const char*)stride_a, *const sb = (const char*)stride_b, *const sc = (const char*)stride_c;
        const float *a0 = (const float*)a, *b0 = (const float*)b, *ai = a0 + da, *bi = b0 + db;
        float *c0 = (float*)c, *ci = c0 + dc;
        for (i = 0; i < end; ++i) {
          const libxsmm_blasint ii = (i + 1) * index_stride;
          const float *const an = a0 + (0 != stride_a ? (*((const libxsmm_blasint*)(sa + ii)) - index_base) : 0);
          const float *const bn = b0 + (0 != stride_b ? (*((const libxsmm_blasint*)(sb + ii)) - index_base) : 0);
          float       *const cn = c0 + (0 != stride_c ? (*((const libxsmm_blasint*)(sc + ii)) - index_base) : 0);
#if defined(LIBXSMM_GEMM_CHECK)
          if (0 != ai && 0 != bi && 0 != ci)
#endif
          {
            libxsmm_blas_sgemm(transa, transb, &m, &n, &k, alpha, ai, lda, bi, ldb, beta, ci, ldc);
          }
          ai = an; bi = bn; ci = cn;
        }
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
    else { /* singular strides are measured in Bytes */
      const libxsmm_blasint stride_unit = sizeof(libxsmm_blasint);
      const libxsmm_blasint da = (0 != stride_a ? (*stride_a - index_base * stride_unit) : 0);
      const libxsmm_blasint db = (0 != stride_b ? (*stride_b - index_base * stride_unit) : 0);
      const libxsmm_blasint dc = (0 != stride_c ? (*stride_c - index_base * stride_unit) : 0);
      if (((int)sizeof(float)) <= LIBXSMM_MIN(LIBXSMM_MIN(da, db), dc)) {
        const char *a0 = (const char*)a, *b0 = (const char*)b, *ai = a0, *bi = b0;
        char *c0 = (char*)c, *ci = c0;
        for (i = 0; i < end; ++i) {
          const char *const an = ai + da;
          const char *const bn = bi + db;
          char *const cn = ci + dc;
#if defined(LIBXSMM_GEMM_CHECK)
          if (0 != *((const float**)ai) && 0 != *((const float**)bi) && 0 != *((const float**)ci))
#endif
          {
            libxsmm_blas_sgemm(transa, transb, &m, &n, &k, alpha, *((const float**)ai), lda, *((const float**)bi), ldb, beta, *((float**)ci), ldc);
          }
          ai = an; bi = bn; ci = cn; /* next */
        }
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
  }
  else { /* incorrect argument(s) */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API void libxsmm_gemm_batch2(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  const int gemm_flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const descriptor = libxsmm_gemm_descriptor_init2(&blob, iprec, oprec, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    0 != ldc ? *ldc : m, alpha, beta, gemm_flags, libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO));
  const libxsmm_xmmfunction kernel = libxsmm_xmmdispatch(descriptor);
  static int error_once = 0;
  int result;

  if (0 != kernel.xmm) {
    result = libxsmm_mmbatch(kernel, index_base, index_stride,
      stride_a, stride_b, stride_c, a, b, c, batchsize,
      0/*tid*/, 1/*nthreads*/);
  }
  else { /* fall-back */
    switch (iprec) {
      case LIBXSMM_GEMM_PRECISION_F64: {
        result = libxsmm_dmmbatch_blas(transa, transb, m, n, k,
          (const double*)alpha, a, lda, b, ldb, (const double*)beta, c, ldc,
          index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
        result = libxsmm_smmbatch_blas(transa, transb, m, n, k,
          (const float*)alpha, a, lda, b, ldb, (const float*)beta, c, ldc,
          index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
      } break;
      default: result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS != result
    && 0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: libxsmm_gemm_batch failed!\n");
  }
}


LIBXSMM_API void libxsmm_gemm_batch(libxsmm_gemm_precision precision,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  libxsmm_gemm_batch2(precision, precision, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
}


#if defined(LIBXSMM_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_wigemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const int*, const short*, const libxsmm_blasint*,
  const short*, const libxsmm_blasint*,
  const int*, int*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_wigemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const int* alpha, const short* a, const libxsmm_blasint* lda,
  const short* b, const libxsmm_blasint* ldb,
  const int* beta, int* c, const libxsmm_blasint* ldc)
{
  libxsmm_wigemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_wsgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const short*, const libxsmm_blasint*,
  const short*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_wsgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const short* a, const libxsmm_blasint* lda,
  const short* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_wsgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_xgemm)(const libxsmm_gemm_precision*, const libxsmm_gemm_precision*,
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_xgemm)(const libxsmm_gemm_precision* iprec, const libxsmm_gemm_precision* oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_ASSERT(NULL != iprec && NULL != oprec);
  libxsmm_blas_xgemm(*iprec, *oprec, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_dgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_sgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_mmbatch)(libxsmm_xmmfunction, const libxsmm_blasint*,
  const libxsmm_blasint*, const libxsmm_blasint[], const libxsmm_blasint[], const libxsmm_blasint[],
  const void*, const void*, void*, const libxsmm_blasint*, const int*, const int*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_mmbatch)(libxsmm_xmmfunction kernel, const libxsmm_blasint* index_base,
  const libxsmm_blasint* index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, const libxsmm_blasint* batchsize, const int* tid, const int* nthreads)
{
  static int error_once = 0;
  LIBXSMM_ASSERT(0 != a && 0 != b && 0 != c && 0 != index_base && 0 != index_stride && 0 != batchsize && 0 != tid && 0 != nthreads);
  if (EXIT_SUCCESS != libxsmm_mmbatch(kernel, *index_base, *index_stride, stride_a, stride_b, stride_c, a, b, c, *batchsize, *tid, *nthreads)
    && 0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: libxsmm_mmbatch failed!\n");
  }
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_gemm_batch)(const libxsmm_gemm_precision*, const libxsmm_gemm_precision*,
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const void*, const void*, const libxsmm_blasint*, const void*, const libxsmm_blasint*,
  const void*, void*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const libxsmm_blasint[], const libxsmm_blasint[], const libxsmm_blasint[],
  const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_gemm_batch)(const libxsmm_gemm_precision* iprec, const libxsmm_gemm_precision* oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* index_base, const libxsmm_blasint* index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint* batchsize)
{
  LIBXSMM_ASSERT(0 != iprec && 0 != oprec && 0 != m && 0 != n && 0 != k && 0 != index_base && 0 != index_stride && 0 != batchsize);
  libxsmm_gemm_batch2(*iprec, *oprec, transa, transb, *m, *n, *k, alpha, a, lda, b, ldb, beta, c, ldc,
    *index_base, *index_stride, stride_a, stride_b, stride_c, *batchsize);
}

#endif /*defined(LIBXSMM_BUILD)*/

