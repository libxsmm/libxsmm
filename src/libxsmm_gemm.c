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
#include <stdlib.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/* min. tile-size is 3x3 rather than 2x2 to avoid remainder tiles of 1x1 */
#if !defined(LIBXSMM_GEMM_TMIN)
# define LIBXSMM_GEMM_TMIN 3
#endif
#if !defined(LIBXSMM_GEMM_NOJIT_TRANS) && \
  /* TODO: fully support calling convention */ \
  (defined(_WIN32) || defined(__CYGWIN__))
# define LIBXSMM_GEMM_NOJIT_TRANS
#endif
#if !defined(LIBXSMM_GEMM_NOJIT_MCOPY) && 0
# define LIBXSMM_GEMM_NOJIT_MCOPY
#endif
#if !defined(LIBXSMM_GEMM_KPARALLELISM) && 0
# define LIBXSMM_GEMM_KPARALLELISM
#endif
#if !defined(LIBXSMM_GEMM_COPY_A)
# define LIBXSMM_GEMM_COPY_A 0
#endif
#if !defined(LIBXSMM_GEMM_COPY_B)
# define LIBXSMM_GEMM_COPY_B 0
#endif
#if !defined(LIBXSMM_GEMM_COPY_C)
# define LIBXSMM_GEMM_COPY_C 0
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

LIBXSMM_APIVAR(unsigned int internal_gemm_tsize_a);
LIBXSMM_APIVAR(unsigned int internal_gemm_tsize_b);
LIBXSMM_APIVAR(unsigned int internal_gemm_tsize_c);
LIBXSMM_APIVAR(char* internal_gemm_scratch_a);
LIBXSMM_APIVAR(char* internal_gemm_scratch_b);
LIBXSMM_APIVAR(char* internal_gemm_scratch_c);


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
  /* setup tile sizes according to CPUID or environment (LIBXSMM_TGEMM_M, LIBXSMM_TGEMM_N, LIBXSMM_TGEMM_K) */
  static unsigned int config_tm[/*config*/][2/*DP/SP*/] = {
    /* generic (hsw) */{ 32, 32 },
    /* mic (knl/knm) */{ 32, 32 },
    /* core (skx)    */{ 32, 32 }
  };
  static unsigned int config_tn[/*config*/][2/*DP/SP*/] = {
    /* generic (hsw) */{ 64, 64 },
    /* mic (knl/knm) */{ 64, 64 },
    /* core (skx)    */{ 64, 64 }
  };
  static unsigned int config_tk[/*config*/][2/*DP/SP*/] = {
    /* generic (hsw) */{ 160, 160 },
    /* mic (knl/knm) */{ 160, 160 },
    /* core (skx)    */{ 160, 160 }
  };
  LIBXSMM_LOCK_ATTR_TYPE(LIBXSMM_GEMM_LOCK) attr;
  unsigned int i;

  LIBXSMM_LOCK_ATTR_INIT(LIBXSMM_GEMM_LOCK, &attr);
  { /* setup prefetch strategy for tiled GEMMs */
    const char *const env_p = getenv("LIBXSMM_TGEMM_PREFETCH");
    const libxsmm_gemm_prefetch_type tiled_prefetch_default = LIBXSMM_GEMM_PREFETCH_AL2_AHEAD;
    const int uid = ((0 == env_p || 0 == *env_p) ? LIBXSMM_PREFETCH_AUTO/*default*/ : atoi(env_p));
    libxsmm_gemm_tiled_prefetch = (0 <= uid ? libxsmm_gemm_uid2prefetch(uid) : tiled_prefetch_default);
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
  { /* load/adjust tile sizes */
    const char *const env_m = getenv("LIBXSMM_TGEMM_M"), *const env_n = getenv("LIBXSMM_TGEMM_N"), *const env_k = getenv("LIBXSMM_TGEMM_K");
    const int m = ((0 == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((0 == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    const int k = ((0 == env_k || 0 == *env_k) ? 0 : atoi(env_k));
    if (LIBXSMM_X86_AVX512_CORE <= archid) {
      libxsmm_gemm_mtile = config_tm[2];
      libxsmm_gemm_ntile = config_tn[2];
      libxsmm_gemm_ktile = config_tk[2];
    }
    else if (LIBXSMM_X86_AVX512_MIC <= archid && LIBXSMM_X86_AVX512_CORE > archid) {
      libxsmm_gemm_mtile = config_tm[1];
      libxsmm_gemm_ntile = config_tn[1];
      libxsmm_gemm_ktile = config_tk[1];
    }
    else {
      libxsmm_gemm_mtile = config_tm[0];
      libxsmm_gemm_ntile = config_tn[0];
      libxsmm_gemm_ktile = config_tk[0];
    }
    { /* double-precision */
      if (0 < m) libxsmm_gemm_mtile[0] = LIBXSMM_MAX(m, LIBXSMM_GEMM_TMIN);
      if (0 < n) libxsmm_gemm_ntile[0] = LIBXSMM_MAX(n, LIBXSMM_GEMM_TMIN);
      if (0 < k) libxsmm_gemm_ktile[0] = LIBXSMM_MAX(k, LIBXSMM_GEMM_TMIN);
      internal_gemm_tsize_a = libxsmm_gemm_mtile[0] * libxsmm_gemm_ktile[0] * 8;
      internal_gemm_tsize_b = libxsmm_gemm_ktile[0] * libxsmm_gemm_ntile[0] * 8;
      internal_gemm_tsize_c = libxsmm_gemm_mtile[0] * libxsmm_gemm_ntile[0] * 8;
    }
    { /* single-precision */
      unsigned int size;
      if (0 < m) libxsmm_gemm_mtile[1] = LIBXSMM_MAX(m, LIBXSMM_GEMM_TMIN);
      if (0 < n) libxsmm_gemm_ntile[1] = LIBXSMM_MAX(n, LIBXSMM_GEMM_TMIN);
      if (0 < k) libxsmm_gemm_ktile[1] = LIBXSMM_MAX(k, LIBXSMM_GEMM_TMIN);
      size = libxsmm_gemm_mtile[1] * libxsmm_gemm_ktile[1] * 4;
      internal_gemm_tsize_a = LIBXSMM_MAX(internal_gemm_tsize_a, size);
      size = libxsmm_gemm_ktile[1] * libxsmm_gemm_ntile[1] * 4;
      internal_gemm_tsize_b = LIBXSMM_MAX(internal_gemm_tsize_b, size);
      size = libxsmm_gemm_mtile[1] * libxsmm_gemm_ntile[1] * 4;
      internal_gemm_tsize_c = LIBXSMM_MAX(internal_gemm_tsize_c, size);
    }
  }
  { /* thread-local scratch buffer for GEMM */
    void* buffer;
    internal_gemm_tsize_a = LIBXSMM_UP2(internal_gemm_tsize_a, LIBXSMM_CACHELINE);
    internal_gemm_tsize_b = LIBXSMM_UP2(internal_gemm_tsize_b, LIBXSMM_CACHELINE);
    internal_gemm_tsize_c = LIBXSMM_UP2(internal_gemm_tsize_c, LIBXSMM_CACHELINE);
    if (EXIT_SUCCESS == libxsmm_xmalloc(&buffer, (size_t)(LIBXSMM_MAX_NTHREADS) * internal_gemm_tsize_a,
      LIBXSMM_CACHELINE, LIBXSMM_MALLOC_FLAG_SCRATCH, NULL/*extra*/, 0/*extra_size*/))
    {
      internal_gemm_scratch_a = (char*)buffer;
    }
    else {
      fprintf(stderr, "LIBXSMM ERROR: failed to allocate GEMM-scratch memory for A-matrix!\n");
    }
    if (EXIT_SUCCESS == libxsmm_xmalloc(&buffer, (size_t)(LIBXSMM_MAX_NTHREADS) * internal_gemm_tsize_b,
      LIBXSMM_CACHELINE, LIBXSMM_MALLOC_FLAG_SCRATCH, NULL/*extra*/, 0/*extra_size*/))
    {
      internal_gemm_scratch_b = (char*)buffer;
    }
    else {
      fprintf(stderr, "LIBXSMM ERROR: failed to allocate GEMM-scratch memory for B-matrix!\n");
    }
    if (EXIT_SUCCESS == libxsmm_xmalloc(&buffer, (size_t)(LIBXSMM_MAX_NTHREADS) * internal_gemm_tsize_c,
      LIBXSMM_CACHELINE, LIBXSMM_MALLOC_FLAG_SCRATCH, NULL/*extra*/, 0/*extra_size*/))
    {
      internal_gemm_scratch_c = (char*)buffer;
    }
    else {
      fprintf(stderr, "LIBXSMM ERROR: failed to allocate GEMM-scratch memory for C-matrix!\n");
    }
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
  libxsmm_free(internal_gemm_scratch_a);
  libxsmm_free(internal_gemm_scratch_b);
  libxsmm_free(internal_gemm_scratch_c);
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
    }
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
    }
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


LIBXSMM_API libxsmm_gemm_handle* libxsmm_gemm_handle_init(libxsmm_gemm_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta, /*unsigned*/int nthreads)
{
  unsigned int ulda, uldb, um, un, uk, bm = 0, bn = 0, bk = 0;
  libxsmm_descriptor_blob desc_blob;
  union {
    libxsmm_gemm_handle* ptr;
    libxsmm_gemm_blob* blob;
  } result;
  LIBXSMM_ASSERT(sizeof(libxsmm_gemm_handle) <= sizeof(libxsmm_gemm_blob));
  if (NULL != blob && NULL != m) {
    unsigned int tmp, rm, rn, rk;
    LIBXSMM_INIT
    result.blob = blob;
    result.ptr->copy_a.pmm = result.ptr->copy_b.pmm = result.ptr->copy_i.pmm = result.ptr->copy_o.pmm = NULL;
    result.ptr->flags_gemm = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
    result.ptr->flags_copy = 0/*LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE*/;
    result.ptr->prf_copy = 0;
    /* TODO: check that arguments fit into handle (unsigned int vs. libxsmm_blasint) */
    uk = (unsigned int)(NULL != k ? *k : *m);
    un = (NULL != n ? ((unsigned int)(*n)) : uk);
    um = (unsigned int)(*m);
    ulda = (NULL != lda ? ((unsigned int)(*lda)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & result.ptr->flags_gemm) ? ((unsigned int)(*m)) : uk));
    uldb = (NULL != ldb ? ((unsigned int)(*ldb)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & result.ptr->flags_gemm) ? uk : un));
    result.ptr->ldc = (unsigned int)(NULL != ldc ? *ldc : *m);
    result.ptr->itypesize = libxsmm_gemm_typesize(iprec);
    result.ptr->otypesize = libxsmm_gemm_typesize(oprec);
    tmp = (4 < result.ptr->otypesize ? 0 : 1);
    bm = LIBXSMM_MIN(libxsmm_gemm_mtile[tmp], um);
    bn = LIBXSMM_MIN(libxsmm_gemm_ntile[tmp], un);
    bk = LIBXSMM_MIN(libxsmm_gemm_ktile[tmp], uk);
    rk = uk % bk;
    rn = un % bn;
    rm = um % bm;
#if 1
    if (0 != rm || 0 != rn || 0 != rk) { /* TODO: implement remainder tiles */
      return NULL;
    }
#endif
    if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->flags_gemm)) { /* NN, NT, or TN */
      result.ptr->tm = bm; result.ptr->tn = bn; result.ptr->tk = bk;
      result.ptr->m = um; result.ptr->n = un; result.ptr->k = uk;
      result.ptr->lda = ulda; result.ptr->ldb = uldb;
      if (0 != (LIBXSMM_GEMM_FLAG_TRANS_A & result.ptr->flags_gemm)) { /* TN */
        const libxsmm_trans_descriptor *const desc = libxsmm_trans_descriptor_init(&desc_blob,
          result.ptr->itypesize, bk, bm, bm/*ldo*/);
        result.ptr->copy_a.xtrans = libxsmm_dispatch_trans(desc);
        if (NULL == result.ptr->copy_a.ptr_const) result.ptr = NULL;
      }
      else if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & result.ptr->flags_gemm)) { /* NT */
        const libxsmm_trans_descriptor *const desc = libxsmm_trans_descriptor_init(&desc_blob,
          result.ptr->itypesize, bn, bk, bk/*ldo*/);
        result.ptr->copy_b.xtrans = libxsmm_dispatch_trans(desc);
        if (NULL == result.ptr->copy_b.ptr_const) result.ptr = NULL;
      }
    }
    else { /* TT */
      const unsigned int mn = LIBXSMM_MIN(bm, bn);
      const libxsmm_trans_descriptor *const desc = libxsmm_trans_descriptor_init(&desc_blob,
        result.ptr->otypesize, mn, mn, result.ptr->ldc/*ldo*/);
      result.ptr->copy_o.xtrans = libxsmm_dispatch_trans(desc);
      result.ptr->tm = mn; result.ptr->tn = mn; result.ptr->tk = bk;
      result.ptr->m = un; result.ptr->n = um; result.ptr->k = uk;
      result.ptr->lda = uldb; result.ptr->ldb = ulda;
      if (NULL != result.ptr->copy_o.ptr_const) {
        double dbeta; /* copy-in only if beta!=0 */
        if (EXIT_SUCCESS == libxsmm_gemm_dvalue(oprec, beta, &dbeta) && LIBXSMM_NEQ(0, dbeta)) {
          result.ptr->copy_i.xtrans = libxsmm_dispatch_trans(libxsmm_trans_descriptor_init(&desc_blob,
            result.ptr->otypesize, mn, mn, mn/*ldo*/));
          if (NULL == result.ptr->copy_i.ptr_const) result.ptr = NULL;
        }
      }
      else {
        result.ptr = NULL;
      }
    }
  }
  else {
    result.ptr = NULL;
  }
#if defined(LIBXSMM_GEMM_COPY_A) && (0 < (LIBXSMM_GEMM_COPY_A))
  if (NULL != result.ptr && NULL == result.ptr->copy_a.ptr_const
# if (1 < (LIBXSMM_GEMM_COPY_A))
    && result.ptr->lda == LIBXSMM_UP2POT(result.ptr->lda)
# endif
  ) {
    const libxsmm_mcopy_descriptor *const desc = LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->flags_gemm)
      ? libxsmm_mcopy_descriptor_init(&desc_blob,
        result.ptr->itypesize, bm, bk, bm/*ldo*/, result.ptr->lda/*ldi*/,
        result.ptr->flags_copy, result.ptr->prf_copy, NULL/*unroll*/)
      : libxsmm_mcopy_descriptor_init(&desc_blob,
        result.ptr->itypesize, LIBXSMM_MIN(bm, bn), bk, bk/*ldo*/, result.ptr->lda/*ldi*/,
        result.ptr->flags_copy, result.ptr->prf_copy, NULL/*unroll*/);
    result.ptr->copy_a.xmatcopy = libxsmm_dispatch_mcopy(desc);
    if (NULL == result.ptr->copy_a.ptr_const) result.ptr = NULL;
  }
#endif
#if defined(LIBXSMM_GEMM_COPY_B) && (0 < (LIBXSMM_GEMM_COPY_B))
  if (NULL != result.ptr && NULL == result.ptr->copy_b.ptr_const
# if (1 < (LIBXSMM_GEMM_COPY_B))
    && result.ptr->ldb == LIBXSMM_UP2POT(result.ptr->ldb)
# endif
  ) {
    const libxsmm_mcopy_descriptor *const desc = libxsmm_mcopy_descriptor_init(&desc_blob,
      result.ptr->itypesize, bk, bn, bk/*ldo*/, result.ptr->ldb/*ldi*/,
      result.ptr->flags_copy, result.ptr->prf_copy, NULL/*unroll*/);
    result.ptr->copy_b.xmatcopy = libxsmm_dispatch_mcopy(desc);
    if (NULL == result.ptr->copy_b.ptr_const) result.ptr = NULL;
  }
#endif
#if defined(LIBXSMM_GEMM_COPY_C) && (0 < (LIBXSMM_GEMM_COPY_C))
  if (NULL != result.ptr && NULL == result.ptr->copy_o.ptr_const
# if (1 < (LIBXSMM_GEMM_COPY_C))
    && result.ptr->ldc == LIBXSMM_UP2POT(result.ptr->ldc)
# endif
  ) {
    result.ptr->copy_o.xmatcopy = libxsmm_dispatch_mcopy(libxsmm_mcopy_descriptor_init(&desc_blob,
      result.ptr->otypesize, bm, bn, result.ptr->ldc/*ldo*/, bm/*ldi*/,
      result.ptr->flags_copy, result.ptr->prf_copy, NULL/*unroll*/));
    if (NULL != result.ptr->copy_o.ptr_const) {
      double dbeta; /* copy-in only if beta!=0 */
      if (EXIT_SUCCESS == libxsmm_gemm_dvalue(oprec, beta, &dbeta) && LIBXSMM_NEQ(0, dbeta)) {
        result.ptr->copy_i.xmatcopy = libxsmm_dispatch_mcopy(libxsmm_mcopy_descriptor_init(&desc_blob,
          result.ptr->otypesize, bm, bn, bm/*ldo*/, result.ptr->ldc/*ldi*/,
          result.ptr->flags_copy, result.ptr->prf_copy, NULL/*unroll*/));
        if (NULL == result.ptr->copy_i.ptr_const) result.ptr = NULL;
      }
    }
    else result.ptr = NULL;
  }
#endif
  if (NULL != result.ptr) { /* build SMM kernels */
    const libxsmm_gemm_prefetch_type prf_gemm = libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
    const libxsmm_blasint klda = (libxsmm_blasint)(NULL == result.ptr->copy_a.ptr_const ? result.ptr->lda : result.ptr->tm);
    const libxsmm_blasint kldb = (libxsmm_blasint)(NULL == result.ptr->copy_b.ptr_const ? result.ptr->ldb : result.ptr->tk);
    const libxsmm_blasint km = (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->flags_gemm) ? result.ptr->tm : result.ptr->tn);
    const libxsmm_blasint kn = (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->flags_gemm) ? result.ptr->tn : result.ptr->tm);
    libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init2( /* remove transpose flags from kernel request */
      &desc_blob, iprec, oprec, km, kn, result.ptr->tk, klda, kldb,
      NULL == result.ptr->copy_o.ptr_const ? ((libxsmm_blasint)result.ptr->ldc) : km,
      alpha, beta, result.ptr->flags_gemm & ~LIBXSMM_GEMM_FLAG_TRANS_AB, prf_gemm);
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
    else result.ptr = NULL;
  }
  if (NULL != result.ptr && 0 < nthreads) { /* plan the work */
    const unsigned int mt = (*m + result.ptr->tm - 1) / result.ptr->tm;
    const unsigned int nt = (result.ptr->n + result.ptr->tn - 1) / result.ptr->tn;
    const unsigned int kt = (result.ptr->k + result.ptr->tk - 1) / result.ptr->tk;
    unsigned int replan = 0;
    result.ptr->nthreads = (unsigned int)nthreads;
    do {
      if (1 >= replan) result.ptr->mt = libxsmm_split_work(mt, result.ptr->nthreads);
      if (1 == replan || result.ptr->nthreads <= result.ptr->mt) { /* M-parallelism */
        result.ptr->nt = 1;
        result.ptr->kt = 1;
        replan = 0;
      }
      else {
        const unsigned int mntasks = libxsmm_split_work(mt * nt, result.ptr->nthreads);
        if (0 == replan && result.ptr->mt >= mntasks) replan = 1;
        if (2 == replan || (0 == replan && result.ptr->nthreads <= mntasks)) { /* MN-parallelism */
          result.ptr->nt = mntasks / result.ptr->mt;
          result.ptr->kt = 1;
          replan = 0;
        }
        else { /* MNK-parallelism */
#if defined(LIBXSMM_GEMM_KPARALLELISM)
          const unsigned int mnktasks = libxsmm_split_work(mt * nt * kt, result.ptr->nthreads);
          if (mntasks < mnktasks) {
            result.ptr->nt = mntasks / result.ptr->mt;
            result.ptr->kt = mnktasks / mntasks;
            replan = 0;
          }
          else
#endif
          if (0 == replan) replan = 2;
        }
      }
    } while (0 != replan);
    result.ptr->dm = mt / result.ptr->mt * result.ptr->tm;
    result.ptr->dn = nt / result.ptr->nt * result.ptr->tn;
    result.ptr->dk = kt / result.ptr->kt * result.ptr->tk;
  }
  return result.ptr;
}


LIBXSMM_API void libxsmm_gemm_thread(const libxsmm_gemm_handle* handle,
  const void* a, const void* b, void* c, /*unsigned*/int tid)
{
#if !defined(NDEBUG)
  if (NULL != handle)
#endif
  {
    const int ntkt = handle->nt * handle->kt, mtid = tid / ntkt, rtid = tid - mtid * ntkt, ntid = rtid / handle->kt, ktid = rtid - ntid * handle->kt;
    const unsigned int m0 = mtid * handle->dm, m1 = LIBXSMM_MIN(m0 + handle->dm, handle->m);
    const unsigned int n0 = ntid * handle->dn, n1 = LIBXSMM_MIN(n0 + handle->dn, handle->n);
    const unsigned int k0 = ktid * handle->dk, k1 = LIBXSMM_MIN(k0 + handle->dk, handle->k);
    const libxsmm_gemm_handle* task = handle;
#if 0
    const int rm = (m1 < (m0 + handle->dm) ? 1 : 0);
    const int rn = (n1 < (n0 + handle->dn) ? 1 : 0);
    const int rk = (k1 < (k0 + handle->dk) ? 1 : 0);
    libxsmm_gemm_handle remainder;
    libxsmm_mmkernel_info info;
    if (EXIT_SUCCESS == libxsmm_get_mmkernel_info(task->kernel[0], &info, NULL/*code_size*/)) {
      const char transa = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & info.flags) ? 'n' : 't');
      const char transb = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & info.flags) ? 'n' : 't');
      const double alpha = 1, beta = 0;
      libxsmm_gemm_handle_init(&remainder, info.iprecision, info.oprecision, &transa, &transb,
        m, n, k, info.lda, info.ldb, info.ldc, &alpha, &beta, 1/*nthreads*/);
      task = &remainder;
    }
#endif
    /* calculate increments to simplify address calculations */
    const unsigned int dom = task->tm * task->otypesize;
    const unsigned int don = task->tn * task->otypesize;
    const unsigned int dik = task->tk * task->itypesize;
    const unsigned int on = task->otypesize * n0;
    unsigned int om = task->otypesize * m0;
    /* calculate base addresses of thread-local storages */
    char *const at = internal_gemm_scratch_a + (size_t)tid * internal_gemm_tsize_a;
    char *const bt = internal_gemm_scratch_b + (size_t)tid * internal_gemm_tsize_b;
    char *const ct = internal_gemm_scratch_c + (size_t)tid * internal_gemm_tsize_c;
    /* loop induction variables */
    unsigned int im = m0, in = n0, ik = k0, im1, in1, ik1;
    LIBXSMM_ASSERT_MSG(mtid * task->dm <= m1 && m1 <= task->m, "Invalid task size!");
    LIBXSMM_ASSERT_MSG(ntid * task->dn <= n1 && n1 <= task->n, "Invalid task size!");
    LIBXSMM_ASSERT_MSG(ktid * task->dk <= k1 && k1 <= task->k, "Invalid task size!");
    for (im1 = im + task->tm; (im1 - 1) < m1; im = im1, im1 += task->tm, om += dom) {
      unsigned int dn = don, dka = dik, dkb = dik;
      char *c0 = (char*)c, *ci;
      const char *aa;
      if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & task->flags_gemm)) {
        if (0 != (LIBXSMM_GEMM_FLAG_TRANS_A & task->flags_gemm)) { /* TN */
          aa = (const char*)a + ((size_t)im * task->lda + k0) * task->itypesize;
        }
        else if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & task->flags_gemm)) { /* NT */
          aa = (const char*)a + ((size_t)k0 * task->lda + im) * task->itypesize;
          dka *= task->lda; dkb *= task->ldb;
        }
        else { /* NN */
          aa = (const char*)a + ((size_t)k0 * task->lda + im) * task->itypesize;
          dka *= task->lda;
        }
        c0 += (size_t)on * task->ldc + om;
        dn *= task->ldc;
      }
      else { /* TT */
        aa = (const char*)b + ((size_t)k0 * task->lda + im) * task->itypesize;
        c0 += (size_t)on + task->ldc * (size_t)om;
        dka *= task->lda;
      }
      for (in = n0, in1 = in + task->tn; (in1 - 1) < n1; in = in1, in1 += task->tn, c0 += dn) {
        const char *a0 = aa, *b0 = (const char*)b;
        if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & task->flags_gemm)) {
          if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & task->flags_gemm)) { /* NT */
            b0 += ((size_t)k0 * task->ldb + in) * task->itypesize;
          }
          else { /* NN or TN */
            b0 += ((size_t)in * task->ldb + k0) * task->itypesize;
          }
        }
        else { /* TT */
          b0 = (const char*)a + ((size_t)in * task->ldb + k0) * task->itypesize;
        }
        if (NULL == task->copy_i.ptr_const) {
          ci = (NULL == task->copy_o.ptr_const ? c0 : ct);
        }
        else {
#if defined(LIBXSMM_GEMM_NOJIT_TRANS)
          if (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & task->flags_gemm)) {
            const unsigned int km = task->tn, kn = task->tm;
            libxsmm_otrans_internal(ct/*out*/, c0/*in*/, task->otypesize, task->ldc/*ldi*/, kn/*ldo*/,
              0, km, 0, kn, km/*tile*/, kn/*tile*/, NULL/*kernel*/);
          }
          else
#endif
#if defined(LIBXSMM_GEMM_NOJIT_MCOPY)
          libxsmm_matcopy_internal(ct/*out*/, c0/*in*/, task->otypesize, task->ldc/*ldi*/, task->tm/*ldo*/,
              0, task->tm, 0, task->tn, task->tm/*tile*/, task->tn/*tile*/, NULL/*kernel*/);
#else
          task->copy_i.xmatcopy(c0, &task->ldc, ct, &task->tm);
#endif
          ci = ct;
        }
        for (ik = k0, ik1 = ik + task->tk; (ik1 - 1) < k1; ik = ik1, ik1 += task->tk) {
          const char *const a1 = a0 + dka, *const b1 = b0 + dkb, *ai = a0, *bi = b0;
          if (NULL != task->copy_a.ptr_const) {
#if defined(LIBXSMM_GEMM_NOJIT_TRANS)
            if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & task->flags_gemm) &&
               (LIBXSMM_GEMM_FLAG_TRANS_A & task->flags_gemm) != 0) /* pure A-transpose */
            {
              libxsmm_otrans_internal(at/*out*/, a0/*in*/, task->itypesize, task->lda/*ldi*/, task->tm/*ldo*/,
                0, task->tk, 0, task->tm, task->tk/*tile*/, task->tm/*tile*/, NULL/*kernel*/);
            }
            else
#endif
#if defined(LIBXSMM_GEMM_NOJIT_MCOPY)
            if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & task->flags_gemm)) {
              libxsmm_matcopy_internal(at/*out*/, a0/*in*/, task->itypesize, task->lda/*ldi*/, task->tm/*ldo*/,
                0, task->tm, 0, task->tk, task->tm/*tile*/, task->tk/*tile*/, NULL/*kernel*/);
            }
            else {
              libxsmm_matcopy_internal(at/*out*/, a0/*in*/, task->itypesize, task->lda/*ldi*/, task->tk/*ldo*/,
                0, task->tm, 0, task->tk, task->tm/*tile*/, task->tk/*tile*/, NULL/*kernel*/);
            }
#else
            task->copy_a.xmatcopy(a0, &task->lda, at, &task->tm);
#endif
            ai = at;
          }
          if (NULL != task->copy_b.ptr_const) {
#if defined(LIBXSMM_GEMM_NOJIT_TRANS)
            if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & task->flags_gemm) &&
               (LIBXSMM_GEMM_FLAG_TRANS_B & task->flags_gemm) != 0) /* pure B-transpose */
            {
              libxsmm_otrans_internal(bt/*out*/, b0/*in*/, task->itypesize, task->ldb/*ldi*/, task->tk/*ldo*/,
                0, task->tn, 0, task->tk, task->tn/*tile*/, task->tk/*tile*/, NULL/*kernel*/);
            }
            else
#endif
#if defined(LIBXSMM_GEMM_NOJIT_MCOPY)
            libxsmm_matcopy_internal(bt/*out*/, b0/*in*/, task->itypesize, task->ldb/*ldi*/, task->tk/*ldo*/,
              0, task->tk, 0, task->tn, task->tk/*tile*/, task->tn/*tile*/, NULL/*kernel*/);
#else
            task->copy_b.xmatcopy(b0, &task->ldb, bt, &task->tk);
#endif
            bi = bt;
          }
          /* beta0-kernel on first-touch, beta1-kernel otherwise (beta0/beta1 are identical if beta=1) */
          LIBXSMM_MMCALL_PRF(task->kernel[k0!=ik?1:0].xmm, ai, bi, ci, a1, b1, c0);
          a0 = a1;
          b0 = b1;
        }
        if (NULL != task->copy_o.ptr_const) { /* TODO: synchronize */
#if defined(LIBXSMM_GEMM_NOJIT_TRANS)
          if (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & task->flags_gemm)) {
            const unsigned int km = task->tm, kn = task->tn;
            libxsmm_otrans_internal(c0/*out*/, ct/*in*/, task->otypesize, km/*ldi*/, task->ldc/*ldo*/,
              0, km, 0, kn, km/*tile*/, kn/*tile*/, NULL/*kernel*/);
          }
          else
#endif
#if defined(LIBXSMM_GEMM_NOJIT_MCOPY)
          libxsmm_matcopy_internal(c0/*out*/, ct/*in*/, task->otypesize, task->tm/*ldi*/, task->ldc/*ldo*/,
            0, task->tm, 0, task->tn, task->tm/*tile*/, task->tn/*tile*/, NULL/*kernel*/);
#else
          task->copy_o.xmatcopy(ct, &task->tm, c0, &task->ldc);
#endif
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

