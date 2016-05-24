/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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
/* Hans Pabst (Intel Corp.), Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_intrinsics_x86.h"
#include "libxsmm_cpuid_x86.h"
#include "libxsmm_gemm_diff.h"
#include "libxsmm_gemm_ext.h"
#include "libxsmm_hash.h"
#include "libxsmm_sync.h"

#if defined(__TRACE)
# include "libxsmm_trace.h"
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
/* mute warning about target attribute; KNC/native plus JIT is disabled below! */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if !defined(NDEBUG)
# include <assert.h>
# include <errno.h>
#endif
#if defined(_WIN32)
# include <Windows.h>
#else
# if !defined(LIBXSMM_INTERNAL_MAP)
#   define LIBXSMM_INTERNAL_MAP MAP_PRIVATE
# endif
# include <sys/mman.h>
# include <pthread.h>
# include <unistd.h>
# include <fcntl.h>
#endif
#if defined(LIBXSMM_VTUNE)
# include <jitprofiling.h>
# define LIBXSMM_VTUNE_JITVERSION 2
# if (2 == LIBXSMM_VTUNE_JITVERSION)
#   define LIBXSMM_VTUNE_JIT_DESC_TYPE iJIT_Method_Load_V2
#   define LIBXSMM_VTUNE_JIT_LOAD iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED_V2
# else
#   define LIBXSMM_VTUNE_JIT_DESC_TYPE iJIT_Method_Load
#   define LIBXSMM_VTUNE_JIT_LOAD iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED
# endif
# define LIBXSMM_VTUNE_JIT_UNLOAD iJVM_EVENT_TYPE_METHOD_UNLOAD_START
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/**
 * LIBXSMM is agnostic with respect to the threading runtime!
 * LIBXSMM_OPENMP suppresses using OS primitives (PThreads)
 */
#if defined(_OPENMP) && !defined(LIBXSMM_OPENMP)
/*# define LIBXSMM_OPENMP*/
#endif

/* alternative hash algorithm (instead of CRC32) */
#if !defined(LIBXSMM_HASH_BASIC) && !defined(LIBXSMM_REGSIZE)
# if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH) || (LIBXSMM_X86_SSE4_2 > LIBXSMM_MAX_STATIC_TARGET_ARCH)
/*#   define LIBXSMM_HASH_BASIC*/
# endif
#endif

/* allow external definition to enable testing corner cases (exhausted registry space) */
#if !defined(LIBXSMM_REGSIZE)
# if defined(LIBXSMM_HASH_BASIC) /* consider larger registry to better deal with low-quality hash */
#   define LIBXSMM_REGSIZE /*1048576*/524288 /* no Mersenne Prime number required, but POT number */
# else
#   define LIBXSMM_REGSIZE 524288 /* 524287: Mersenne Prime number (2^19-1) */
# endif
# define LIBXSMM_HASH_MOD(N, NPOT) LIBXSMM_MOD2(N, NPOT)
#else
# define LIBXSMM_HASH_MOD(N, NGEN) ((N) % (NGEN))
#endif

#if !defined(LIBXSMM_CACHESIZE)
# define LIBXSMM_CACHESIZE 4
#endif

#if defined(LIBXSMM_HASH_BASIC)
# define LIBXSMM_HASH_FUNCTION_CALL(HASH, INDX, DESCRIPTOR) \
    HASH = libxsmm_hash_npot(&(DESCRIPTOR), LIBXSMM_GEMM_DESCRIPTOR_SIZE, LIBXSMM_REGSIZE); \
    assert((LIBXSMM_REGSIZE) > (HASH)); \
    INDX = (HASH)
#else
# define LIBXSMM_HASH_FUNCTION_CALL(HASH, INDX, DESCRIPTOR) \
    HASH = libxsmm_crc32(&(DESCRIPTOR), LIBXSMM_GEMM_DESCRIPTOR_SIZE, 25071975/*seed*/); \
    INDX = LIBXSMM_HASH_MOD(HASH, LIBXSMM_REGSIZE)
#endif

/* flag fused into the memory address of a code version in case of collision */
#define LIBXSMM_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 1))

#if 16 >= (LIBXSMM_GEMM_DESCRIPTOR_SIZE)
# define LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE 16
#elif 32 >= (LIBXSMM_GEMM_DESCRIPTOR_SIZE)
# define LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE 32
#else
# define LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE LIBXSMM_GEMM_DESCRIPTOR_SIZE
#endif

typedef union LIBXSMM_RETARGETABLE internal_regkey {
  char simd[LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE];
  libxsmm_gemm_descriptor descriptor;
} internal_regkey;

typedef struct LIBXSMM_RETARGETABLE internal_regentry {
  union {
    libxsmm_xmmfunction xmm;
    /*const*/void* pmm;
    uintptr_t imm;
  } function;
  /* statically generated code (=0), dynamically generated code (>0). */
  unsigned int size;
#if defined(LIBXSMM_VTUNE)
  unsigned int id;
#endif
} internal_regentry;

LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL struct LIBXSMM_RETARGETABLE {
  unsigned int ntry, ncol, njit, nsta;
} internal_statistic[2/*DP/SP*/][3/*sml/med/big*/];

LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL unsigned int internal_statistic_sml = 13;
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL unsigned int internal_statistic_med = 23;
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL unsigned int internal_statistic_mnk = LIBXSMM_MAX_M;

#if defined(NDEBUG)
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL int internal_verbose = 0; /* quiet */
#else
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL int internal_verbose = 1; /* verbose */
#endif

LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL internal_regkey* internal_registry_keys = 0;
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL internal_regentry* internal_registry = 0;
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL unsigned int internal_teardown = 0;

/** Helper macro determining the default prefetch strategy which is used for statically generated kernels. */
#if defined(_WIN32) || defined(__CYGWIN__) /*TODO: account for calling convention; avoid passing six arguments*/
# define INTERNAL_PREFETCH LIBXSMM_PREFETCH_NONE
#elif defined(__MIC__) && (0 > LIBXSMM_PREFETCH) /* auto-prefetch (frontend) */
# define INTERNAL_PREFETCH LIBXSMM_PREFETCH_AL2BL2_VIA_C
#elif (0 > LIBXSMM_PREFETCH) /* auto-prefetch (frontend) */
# define INTERNAL_PREFETCH LIBXSMM_PREFETCH_SIGONLY
#endif
#if !defined(INTERNAL_PREFETCH)
# define INTERNAL_PREFETCH LIBXSMM_PREFETCH
#endif

LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL int internal_prefetch = LIBXSMM_MAX(INTERNAL_PREFETCH, 0);
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL int internal_target_archid = LIBXSMM_TARGET_ARCH_GENERIC;

#if !defined(LIBXSMM_OPENMP)
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL LIBXSMM_LOCK_TYPE internal_reglock[] = {
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT
};
#endif

#if defined(__GNUC__)
# define LIBXSMM_INIT
  /* libxsmm_init already executed via GCC constructor attribute */
# define INTERNAL_FIND_CODE_INIT(VARIABLE) assert(0 != (VARIABLE))
#else /* lazy initialization */
# define LIBXSMM_INIT libxsmm_init();
  /* use return value of internal_init to refresh local representation */
# define INTERNAL_FIND_CODE_INIT(VARIABLE) if (0 == (VARIABLE)) VARIABLE = internal_init()
#endif

#if defined(LIBXSMM_OPENMP)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX) LIBXSMM_PRAGMA(omp critical(internal_reglock)) { \
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) }
#else
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX) { \
    const unsigned int LOCKINDEX = LIBXSMM_MOD2(INDEX, sizeof(internal_reglock) / sizeof(*internal_reglock)); \
    LIBXSMM_LOCK_ACQUIRE(internal_reglock[LOCKINDEX])
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXSMM_LOCK_RELEASE(internal_reglock[LOCKINDEX]); }
#endif

#define INTERNAL_FIND_CODE_DECLARE(CODE) internal_regentry* CODE = LIBXSMM_ATOMIC_LOAD(internal_registry, LIBXSMM_ATOMIC_RELAXED); unsigned int i
#define INTERNAL_FIND_CODE_READ(CODE, DST) DST = LIBXSMM_ATOMIC_LOAD((CODE)->function.pmm, LIBXSMM_ATOMIC_SEQ_CST)
#define INTERNAL_FIND_CODE_WRITE(CODE, SRC) LIBXSMM_ATOMIC_STORE((CODE)->function.pmm, SRC, LIBXSMM_ATOMIC_SEQ_CST)

#if defined(LIBXSMM_CACHESIZE) && (0 < (LIBXSMM_CACHESIZE))
# define INTERNAL_FIND_CODE_CACHE_DECL(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT) \
  static LIBXSMM_TLS union { libxsmm_gemm_descriptor desc; char padding[LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE]; } CACHE_KEYS[LIBXSMM_CACHESIZE]; \
  static LIBXSMM_TLS libxsmm_xmmfunction CACHE[LIBXSMM_CACHESIZE]; \
  static LIBXSMM_TLS unsigned int CACHE_ID = (unsigned int)(-1); \
  static LIBXSMM_TLS unsigned int CACHE_HIT = LIBXSMM_CACHESIZE;
# define INTERNAL_FIND_CODE_CACHE_BEGIN(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR) \
  assert(LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE >= LIBXSMM_GEMM_DESCRIPTOR_SIZE); \
  /* search small cache starting with the last hit on record */ \
  i = libxsmm_gemm_diffn(DESCRIPTOR, &(CACHE_KEYS)->desc, CACHE_HIT, LIBXSMM_CACHESIZE, LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE); \
  if ((LIBXSMM_CACHESIZE) > i && (CACHE_ID) == internal_teardown) { /* cache hit, and valid */ \
    (RESULT).function.xmm = (CACHE)[i]; \
    CACHE_HIT = i; \
  } \
  else
# if defined(LIBXSMM_GEMM_DIFF_SW) && (2 == (LIBXSMM_GEMM_DIFF_SW)) /* most general implementation */
#   define INTERNAL_FIND_CODE_CACHE_FINALIZE(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR) \
    if ((CACHE_ID) != internal_teardown) { \
      memset(CACHE_KEYS, -1, sizeof(CACHE_KEYS)); \
      CACHE_ID = internal_teardown; \
    } \
    i = ((CACHE_HIT) + ((LIBXSMM_CACHESIZE) - 1)) % (LIBXSMM_CACHESIZE); \
    ((CACHE_KEYS)[i]).desc = *(DESCRIPTOR); \
    (CACHE)[i] = (RESULT).function.xmm; \
    CACHE_HIT = i
# else
#   define INTERNAL_FIND_CODE_CACHE_FINALIZE(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR) \
    assert(/*is pot*/(LIBXSMM_CACHESIZE) == (1 << LIBXSMM_LOG2(LIBXSMM_CACHESIZE))); \
    if ((CACHE_ID) != internal_teardown) { \
      memset(CACHE_KEYS, -1, sizeof(CACHE_KEYS)); \
      CACHE_ID = internal_teardown; \
    } \
    i = LIBXSMM_MOD2((CACHE_HIT) + ((LIBXSMM_CACHESIZE) - 1), LIBXSMM_CACHESIZE); \
    (CACHE_KEYS)[i].desc = *(DESCRIPTOR); \
    (CACHE)[i] = (RESULT).function.xmm; \
    CACHE_HIT = i
# endif
#else
# define INTERNAL_FIND_CODE_CACHE_DECL(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT)
# define INTERNAL_FIND_CODE_CACHE_BEGIN(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR)
# define INTERNAL_FIND_CODE_CACHE_FINALIZE(CACHE_ID, CACHE_KEYS, CACHE, CACHE_HIT, RESULT, DESCRIPTOR)
#endif

#if (0 != LIBXSMM_JIT)
# define INTERNAL_FIND_CODE_JIT(DESCRIPTOR, CODE, RESULT) \
  /* check if code generation or fix-up is needed, also check whether JIT is supported (CPUID) */ \
  if (0 == (RESULT).function.pmm /* code version does not exist */ && LIBXSMM_X86_AVX <= internal_target_archid) { \
    /* instead of blocking others, a try-lock would allow to let others to fallback to BLAS (return 0) during lock-time */ \
    INTERNAL_FIND_CODE_LOCK(lock, i); /* lock the registry entry */ \
    /* re-read registry entry after acquiring the lock */ \
    if (0 == diff) { \
      RESULT = *(CODE); \
      (RESULT).function.imm &= ~LIBXSMM_HASH_COLLISION; \
    } \
    if (0 == (RESULT).function.pmm) { /* double-check after acquiring the lock */ \
      if (0 == diff) { \
        /* found a conflict-free registry-slot, and attempt to build the kernel */ \
        internal_build(DESCRIPTOR, &(RESULT)); \
        internal_update_statistic(DESCRIPTOR, 1, 0); \
        if (0 != (RESULT).function.pmm) { /* synchronize registry entry */ \
          internal_registry_keys[i].descriptor = *(DESCRIPTOR); \
          *(CODE) = RESULT; \
          INTERNAL_FIND_CODE_WRITE(CODE, (RESULT).function.pmm); \
        } \
      } \
      else { /* 0 != diff */ \
        if (0 == diff0) { \
          /* flag existing entry as collision */ \
          /*const*/ void * /*const*/ collision = (void*)((CODE)->function.imm | LIBXSMM_HASH_COLLISION); \
          /* find new slot to store the code version */ \
          const unsigned int index = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_REGSIZE); \
          i = (index != i ? index : LIBXSMM_HASH_MOD(index + 1, LIBXSMM_REGSIZE)); \
          i0 = i; /* keep starting point of free-slot-search in mind */ \
          internal_update_statistic(DESCRIPTOR, 0, 1); \
          INTERNAL_FIND_CODE_WRITE(CODE, collision); /* fix-up existing entry */ \
          diff0 = diff; /* no more fix-up */ \
        } \
        else { \
          const unsigned int next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE); \
          if (next != i0) { /* linear search for free slot */ \
            i = next; \
          } \
          else { /* out of registry capacity (no free slot found) */ \
            diff = 0; \
          } \
        } \
        (CODE) = internal_registry + i; \
      } \
    } \
    INTERNAL_FIND_CODE_UNLOCK(lock); \
  } \
  else
#else
# define INTERNAL_FIND_CODE_JIT(DESCRIPTOR, CODE, RESULT)
#endif

#define INTERNAL_FIND_CODE(DESCRIPTOR, CODE) \
  internal_regentry flux_entry; \
{ \
  INTERNAL_FIND_CODE_CACHE_DECL(cache_id, cache_keys, cache, cache_hit) \
  unsigned int hash, diff = 0, diff0 = 0, i0; \
  INTERNAL_FIND_CODE_INIT(CODE); \
  INTERNAL_FIND_CODE_CACHE_BEGIN(cache_id, cache_keys, cache, cache_hit, flux_entry, DESCRIPTOR) { \
    /* check if the requested xGEMM is already JITted */ \
    LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */ \
    LIBXSMM_HASH_FUNCTION_CALL(hash, i = i0, *(DESCRIPTOR)); \
    (CODE) += i; /* actual entry */ \
    do { \
      INTERNAL_FIND_CODE_READ(CODE, flux_entry.function.pmm); /* read registered code */ \
      if (0 != flux_entry.function.pmm) { /* code version exists */ \
        if (0 == diff0) { \
          if (0 == (LIBXSMM_HASH_COLLISION & flux_entry.function.imm)) { /* check for no collision */ \
            /* calculate bitwise difference (deep check) */ \
            LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */ \
            diff = libxsmm_gemm_diff(DESCRIPTOR, &internal_registry_keys[i].descriptor); \
            if (0 != diff) { /* new collision discovered (but no code version yet) */ \
              /* allow to fix-up current entry inside of the guarded/locked region */ \
              flux_entry.function.pmm = 0; \
            } \
          } \
          /* collision discovered but code version exists; perform deep check */ \
          else if (0 != libxsmm_gemm_diff(DESCRIPTOR, &internal_registry_keys[i].descriptor)) { \
            /* continue linearly searching code starting at re-hashed index position */ \
            const unsigned int index = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_REGSIZE); \
            unsigned int next; \
            for (i0 = (index != i ? index : LIBXSMM_HASH_MOD(index + 1, LIBXSMM_REGSIZE)), \
              i = i0, next = LIBXSMM_HASH_MOD(i0 + 1, LIBXSMM_REGSIZE); \
              /* skip any (still invalid) descriptor which corresponds to no code, or continue on difference */ \
              (0 == (CODE = (internal_registry + i))->function.pmm || \
                0 != (diff = libxsmm_gemm_diff(DESCRIPTOR, &internal_registry_keys[i].descriptor))) \
                /* entire registry was searched and no code version was found */ \
                && next != i0; \
              i = next, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE)); \
            if (0 == diff) { /* found exact code version; continue with atomic load */ \
              flux_entry.function.pmm = (CODE)->function.pmm; \
              /* clear the uppermost bit of the address */ \
              flux_entry.function.imm &= ~LIBXSMM_HASH_COLLISION; \
            } \
            else { /* no code found */ \
              flux_entry.function.pmm = 0; \
            } \
            break; \
          } \
          else { /* clear the uppermost bit of the address */ \
            flux_entry.function.imm &= ~LIBXSMM_HASH_COLLISION; \
          } \
        } \
        else { /* new collision discovered (but no code version yet) */ \
          flux_entry.function.pmm = 0; \
        } \
      } \
      INTERNAL_FIND_CODE_JIT(DESCRIPTOR, CODE, flux_entry) \
      { \
        diff = 0; \
      } \
    } \
    while (0 != diff); \
    assert(0 == diff || 0 == flux_entry.function.pmm); \
    INTERNAL_FIND_CODE_CACHE_FINALIZE(cache_id, cache_keys, cache, cache_hit, flux_entry, DESCRIPTOR); \
  } \
} \
return flux_entry.function.xmm

#define INTERNAL_DISPATCH_MAIN(DESCRIPTOR_DECL, DESC, FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/) { \
  INTERNAL_FIND_CODE_DECLARE(code); \
  const signed char scalpha = (signed char)(0 == (PALPHA) ? LIBXSMM_ALPHA : *(PALPHA)), scbeta = (signed char)(0 == (PBETA) ? LIBXSMM_BETA : *(PBETA)); \
  if (0 == ((FLAGS) & (LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B)) && 1 == scalpha && (1 == scbeta || 0 == scbeta)) { \
    const int internal_dispatch_main_prefetch = (0 == (PREFETCH) ? INTERNAL_PREFETCH : *(PREFETCH)); \
    DESCRIPTOR_DECL; LIBXSMM_GEMM_DESCRIPTOR(*(DESC), 0 != (VECTOR_WIDTH) ? (VECTOR_WIDTH): LIBXSMM_ALIGNMENT, FLAGS, LIBXSMM_LD(M, N), LIBXSMM_LD(N, M), K, \
      0 == LIBXSMM_LD(PLDA, PLDB) ? LIBXSMM_LD(M, N) : *LIBXSMM_LD(PLDA, PLDB), \
      0 == LIBXSMM_LD(PLDB, PLDA) ? (K) : *LIBXSMM_LD(PLDB, PLDA), \
      0 == (PLDC) ? LIBXSMM_LD(M, N) : *(PLDC), scalpha, scbeta, \
      0 > internal_dispatch_main_prefetch ? internal_prefetch : internal_dispatch_main_prefetch); \
    { \
      INTERNAL_FIND_CODE(DESC, code).SELECTOR; \
    } \
  } \
  else { /* TODO: not supported (bypass) */ \
    return 0; \
  } \
}

#if defined(LIBXSMM_GEMM_DIFF_MASK_A) /* no padding i.e., LIBXSMM_GEMM_DESCRIPTOR_SIZE */
# define INTERNAL_DISPATCH(FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/) \
    INTERNAL_DISPATCH_MAIN(libxsmm_gemm_descriptor descriptor, &descriptor, \
    FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/)
#else /* padding: LIBXSMM_GEMM_DESCRIPTOR_SIZE -> LIBXSMM_ALIGNMENT */
# define INTERNAL_DISPATCH(FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/) { \
    INTERNAL_DISPATCH_MAIN(union { libxsmm_gemm_descriptor desc; char simd[LIBXSMM_ALIGNMENT]; } simd_descriptor; \
      for (i = LIBXSMM_GEMM_DESCRIPTOR_SIZE; i < sizeof(simd_descriptor.simd); ++i) simd_descriptor.simd[i] = 0, &simd_descriptor.desc, \
    FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/)
#endif

#define INTERNAL_SMMDISPATCH(PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) \
  INTERNAL_DISPATCH((0 == (PFLAGS) ? LIBXSMM_FLAGS : *(PFLAGS)) | LIBXSMM_GEMM_FLAG_F32PREC, \
  M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, smm)
#define INTERNAL_DMMDISPATCH(PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) \
  INTERNAL_DISPATCH((0 == (PFLAGS) ? LIBXSMM_FLAGS : *(PFLAGS)), \
  M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, dmm)


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_update_statistic(const libxsmm_gemm_descriptor* desc,
  unsigned ntry, unsigned ncol)
{
  assert(0 != desc);
  {
    const unsigned long long size = LIBXSMM_MNK_SIZE(desc->m, desc->n, desc->k);
    const int precision = (0 == (LIBXSMM_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1);
    int bucket = 2/*big*/;

    if (LIBXSMM_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= size) {
      bucket = 0;
    }
    else if (LIBXSMM_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= size) {
      bucket = 1;
    }

    LIBXSMM_ATOMIC_ADD_FETCH(internal_statistic[precision][bucket].ntry, ntry, LIBXSMM_ATOMIC_RELAXED);
    LIBXSMM_ATOMIC_ADD_FETCH(internal_statistic[precision][bucket].ncol, ncol, LIBXSMM_ATOMIC_RELAXED);
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE const char* internal_get_target_arch(int id);
LIBXSMM_INLINE LIBXSMM_RETARGETABLE const char* internal_get_target_arch(int id)
{
  const char* target_arch = 0;
  switch (id) {
    case LIBXSMM_X86_AVX512_CORE: {
      target_arch = "skx";
    } break;
    case LIBXSMM_X86_AVX512_MIC: {
      target_arch = "knl";
    } break;
    case LIBXSMM_X86_AVX2: {
      target_arch = "hsw";
    } break;
    case LIBXSMM_X86_AVX: {
      target_arch = "snb";
    } break;
    case LIBXSMM_X86_SSE4_2: {
      target_arch = "wsm";
    } break;
    case LIBXSMM_X86_SSE4_1: {
      target_arch = "sse4";
    } break;
    case LIBXSMM_X86_SSE3: {
      target_arch = "sse3";
    } break;
    case LIBXSMM_TARGET_ARCH_GENERIC: {
      target_arch = "generic";
    } break;
    default: if (LIBXSMM_X86_GENERIC <= id) {
      target_arch = "x86";
    }
    else {
      target_arch = "unknown";
    }
  }

  assert(0 != target_arch);
  return target_arch;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE unsigned int internal_print_statistic(FILE* ostream, const char* target_arch, int precision, unsigned int linebreaks, unsigned int indent)
{
  int printed = 0;
  assert(0 != ostream && 0 != target_arch && (0 <= precision && precision < 2));

  if (/* omit to print anything if it is superfluous */
    0 != internal_statistic[precision][0/*SML*/].ntry ||
    0 != internal_statistic[precision][0/*SML*/].njit ||
    0 != internal_statistic[precision][0/*SML*/].nsta ||
    0 != internal_statistic[precision][0/*SML*/].ncol ||
    0 != internal_statistic[precision][1/*MED*/].ntry ||
    0 != internal_statistic[precision][1/*MED*/].njit ||
    0 != internal_statistic[precision][1/*MED*/].nsta ||
    0 != internal_statistic[precision][1/*MED*/].ncol ||
    0 != internal_statistic[precision][2/*BIG*/].ntry ||
    0 != internal_statistic[precision][2/*BIG*/].njit ||
    0 != internal_statistic[precision][2/*BIG*/].nsta ||
    0 != internal_statistic[precision][2/*BIG*/].ncol)
  {
    char title[256], sml[256], med[256], big[256];

    LIBXSMM_SNPRINTF(sml, sizeof(sml), "%u..%u",                          0u, internal_statistic_sml);
    LIBXSMM_SNPRINTF(med, sizeof(sml), "%u..%u", internal_statistic_sml + 1u, internal_statistic_med);
    LIBXSMM_SNPRINTF(big, sizeof(sml), "%u..%u", internal_statistic_med + 1u, internal_statistic_mnk);
    {
      unsigned int n = 0;
      for (n = 0; 0 != target_arch[n] && n < sizeof(title); ++n) { /* toupper */
        const char c = target_arch[n];
        title[n] = (char)(('a' <= c && c <= 'z') ? (c - 32) : c);
      }
      LIBXSMM_SNPRINTF(title + n, sizeof(title) - n, "/%s", 0 == precision ? "DP" : "SP");
      for (n = 0; n < linebreaks; ++n) fprintf(ostream, "\n");
    }
    fprintf(ostream, "%*s%-10s %6s %6s %6s %6s\n", (int)indent, "", title, "TRY" ,"JIT", "STA", "COL");
    fprintf(ostream,  "%*s%10s %6u %6u %6u %6u\n", (int)indent, "", sml,
      internal_statistic[precision][0/*SML*/].ntry,
      internal_statistic[precision][0/*SML*/].njit,
      internal_statistic[precision][0/*SML*/].nsta,
      internal_statistic[precision][0/*SML*/].ncol);
    fprintf(ostream,  "%*s%10s %6u %6u %6u %6u\n", (int)indent, "", med,
      internal_statistic[precision][1/*MED*/].ntry,
      internal_statistic[precision][1/*MED*/].njit,
      internal_statistic[precision][1/*MED*/].nsta,
      internal_statistic[precision][1/*MED*/].ncol);
    fprintf(ostream,  "%*s%10s %6u %6u %6u %6u\n", (int)indent, "", big,
      internal_statistic[precision][2/*BIG*/].ntry,
      internal_statistic[precision][2/*BIG*/].njit,
      internal_statistic[precision][2/*BIG*/].nsta,
      internal_statistic[precision][2/*BIG*/].ncol);
    printed = 1;
  }

  return printed;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_register_static_code(const libxsmm_gemm_descriptor* desc,
  unsigned int index, unsigned int hash, libxsmm_xmmfunction src, internal_regentry* registry)
{
  internal_regkey* dst_key = internal_registry_keys + index;
  internal_regentry* dst_entry = registry + index;
  assert(0 != desc && 0 != src.dmm && 0 != dst_key && 0 != registry);

  if (0 != dst_entry->function.pmm) { /* collision? */
    /* start at a re-hashed index position */
    const unsigned int start = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_REGSIZE);
    unsigned int i0, i, next;

    /* mark current entry as a collision (this might be already the case) */
    dst_entry->function.imm |= LIBXSMM_HASH_COLLISION;

    /* start linearly searching for an available slot */
    for (i = (start != index) ? start : LIBXSMM_HASH_MOD(start + 1, LIBXSMM_REGSIZE), i0 = i, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE);
      0 != (dst_entry = registry + i)->function.pmm && next != i0; i = next, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE));

    /* corresponding key position */
    dst_key = internal_registry_keys + i;

    internal_update_statistic(desc, 0, 1);
  }

  if (0 == dst_entry->function.pmm) { /* registry not (yet) exhausted */
    dst_entry->function.xmm = src;
    dst_entry->size = 0; /* statically generated code */
    dst_key->descriptor = *desc;
  }

  internal_update_statistic(desc, 1, 0);
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE int internal_get_prefetch(const libxsmm_gemm_descriptor* desc)
{
  assert(0 != desc);
  switch (desc->prefetch) {
    case LIBXSMM_PREFETCH_SIGONLY:            return 2;
    case LIBXSMM_PREFETCH_BL2_VIA_C:          return 3;
    case LIBXSMM_PREFETCH_AL2:                return 4;
    case LIBXSMM_PREFETCH_AL2_AHEAD:          return 5;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C:       return 6;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD: return 7;
    case LIBXSMM_PREFETCH_AL2_JPST:           return 8;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST:  return 9;
    default: {
      assert(LIBXSMM_PREFETCH_NONE == desc->prefetch);
      return 0;
    }
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_get_code_name(const char* target_arch,
  const libxsmm_gemm_descriptor* desc, unsigned int buffer_size, char* name)
{
  assert((0 != desc && 0 != name) || 0 == buffer_size);
  LIBXSMM_SNPRINTF(name, buffer_size, "libxsmm_%s_%c%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.jit",
    target_arch /* code path name */,
    0 == (LIBXSMM_GEMM_FLAG_F32PREC & desc->flags) ? 'd' : 's',
    0 == (LIBXSMM_GEMM_FLAG_TRANS_A & desc->flags) ? 'n' : 't',
    0 == (LIBXSMM_GEMM_FLAG_TRANS_B & desc->flags) ? 'n' : 't',
    (unsigned int)desc->m, (unsigned int)desc->n, (unsigned int)desc->k,
    (unsigned int)desc->lda, (unsigned int)desc->ldb, (unsigned int)desc->ldc,
    desc->alpha, desc->beta, internal_get_prefetch(desc));
}


#if defined(LIBXSMM_VTUNE)
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_get_vtune_jitdesc(const internal_regentry* code, const char* name, LIBXSMM_VTUNE_JIT_DESC_TYPE* desc)
{
  assert(0 != code && 0 != code->id && 0 != code->size && 0 != desc);
  desc->method_id = code->id;
  /* incorrect constness (method_name) */
  desc->method_name = (char*)name;
  desc->method_load_address = code->function.pmm;
  desc->method_size = code->size;
  desc->line_number_size = 0;
  desc->line_number_table = NULL;
  desc->class_file_name = NULL;
  desc->source_file_name = NULL;
# if (2 == LIBXSMM_VTUNE_JITVERSION)
  desc->module_name = "libxsmm.jit";
# endif
}
#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE internal_regentry* internal_init(void)
{
  /*const*/internal_regentry* result;
  int i;

#if !defined(LIBXSMM_OPENMP)
  /* acquire locks and thereby shortcut lazy initialization later on */
  const int nlocks = sizeof(internal_reglock) / sizeof(*internal_reglock);
  for (i = 0; i < nlocks; ++i) LIBXSMM_LOCK_ACQUIRE(internal_reglock[i]);
#else
# pragma omp critical(internal_reglock)
#endif
  {
    result = LIBXSMM_ATOMIC_LOAD(internal_registry, LIBXSMM_ATOMIC_SEQ_CST);
    if (0 == result) {
      int init_code;
      /* set internal_target_archid */
      libxsmm_set_target_arch(getenv("LIBXSMM_TARGET"));
      { /* select prefetch strategy for JIT */
        const char *const env_prefetch = getenv("LIBXSMM_PREFETCH");
        if (0 == env_prefetch || 0 == *env_prefetch) {
          if (0 > LIBXSMM_PREFETCH) { /* permitted by LIBXSMM_PREFETCH_AUTO */
            internal_prefetch = LIBXSMM_X86_AVX512_MIC != internal_target_archid
              ? LIBXSMM_PREFETCH_NONE : LIBXSMM_PREFETCH_AL2BL2_VIA_C;
          }
        }
        else { /* user input considered even if LIBXSMM_PREFETCH_AUTO is disabled */
          switch (atoi(env_prefetch)) {
            case 2: internal_prefetch = LIBXSMM_PREFETCH_SIGONLY; break;
            case 3: internal_prefetch = LIBXSMM_PREFETCH_BL2_VIA_C; break;
            case 4: internal_prefetch = LIBXSMM_PREFETCH_AL2; break;
            case 5: internal_prefetch = LIBXSMM_PREFETCH_AL2_AHEAD; break;
            case 6: internal_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C; break;
            case 7: internal_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD; break;
            case 8: internal_prefetch = LIBXSMM_PREFETCH_AL2_JPST; break;
            case 9: internal_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST; break;
            default: internal_prefetch = LIBXSMM_PREFETCH_NONE;
          }
        }
      }
      libxsmm_hash_init(internal_target_archid);
      libxsmm_gemm_diff_init(internal_target_archid);
      init_code = libxsmm_gemm_init(internal_target_archid, internal_prefetch);
#if defined(__TRACE)
      {
        const char *const env_trace_init = getenv("LIBXSMM_TRACE");
        if (EXIT_SUCCESS == init_code && 0 != env_trace_init) {
          int match[] = { 0, 0 }, filter_threadid = 0, filter_mindepth = 1, filter_maxnsyms = -1;
          char buffer[32];

          if (1 == sscanf(env_trace_init, "%32[^,],", buffer)) {
            sscanf(buffer, "%i", &filter_threadid);
          }
          if (1 == sscanf(env_trace_init, "%*[^,],%32[^,],", buffer)) {
            match[0] = sscanf(buffer, "%i", &filter_mindepth);
          }
          if (1 == sscanf(env_trace_init, "%*[^,],%*[^,],%32s", buffer)) {
            match[1] = sscanf(buffer, "%i", &filter_maxnsyms);
          }
          init_code = (0 == filter_threadid && 0 == match[0] && 0 == match[1]) ? EXIT_SUCCESS
            : libxsmm_trace_init(filter_threadid - 1, filter_mindepth, filter_maxnsyms);
        }
      }
#endif
      if (EXIT_SUCCESS == init_code) {
        assert(0 == internal_registry_keys && 0 == internal_registry/*should never happen*/);
        result = (internal_regentry*)malloc(LIBXSMM_REGSIZE * sizeof(internal_regentry));
        internal_registry_keys = (internal_regkey*)malloc(LIBXSMM_REGSIZE * sizeof(internal_regkey));
        if (result && internal_registry_keys) {
          const char *const env_verbose = getenv("LIBXSMM_VERBOSE");
          internal_statistic_mnk = (unsigned int)(pow((double)(LIBXSMM_MAX_MNK), 0.3333333333333333) + 0.5);
          if (0 != env_verbose && 0 != *env_verbose) {
            internal_verbose = atoi(env_verbose);
          }
          for (i = 0; i < LIBXSMM_REGSIZE; ++i) result[i].function.pmm = 0;
          /* omit registering code if JIT is enabled and if an ISA extension is found
           * which is beyond the static code path used to compile the library
           */
#if (0 != LIBXSMM_JIT) && !defined(__MIC__)
          if (LIBXSMM_STATIC_TARGET_ARCH <= internal_target_archid && LIBXSMM_X86_AVX > internal_target_archid)
#endif
          { /* opening a scope for eventually declaring variables */
            /* setup the dispatch table for the statically generated code */
#           include <libxsmm_dispatch.h>
          }
          atexit(libxsmm_finalize);
          LIBXSMM_ATOMIC_STORE(internal_registry, result, LIBXSMM_ATOMIC_SEQ_CST);
        }
        else {
#if !defined(NDEBUG) && defined(__TRACE) /* library code is expected to be mute */
          fprintf(stderr, "LIBXSMM: failed to allocate code registry!\n");
#endif
          free(internal_registry_keys);
          free(result);
        }
      }
#if !defined(NDEBUG) && defined(__TRACE) /* library code is expected to be mute */
      else {
        fprintf(stderr, "LIBXSMM: failed to initialize sub-component (error #%i)!\n", init_code);
      }
#endif
    }
  }
#if !defined(LIBXSMM_OPENMP) /* release locks */
  for (i = 0; i < nlocks; ++i) LIBXSMM_LOCK_RELEASE(internal_reglock[i]);
#endif
  assert(result);
  return result;
}


LIBXSMM_EXTERN_C
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(constructor)
#endif
LIBXSMM_RETARGETABLE void libxsmm_init(void)
{
  const void *const registry = LIBXSMM_ATOMIC_LOAD(internal_registry, LIBXSMM_ATOMIC_RELAXED);
  if (0 == registry) {
    internal_init();
  }
}


LIBXSMM_EXTERN_C
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(destructor)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
LIBXSMM_RETARGETABLE void libxsmm_finalize(void)
{
  internal_regentry* registry = LIBXSMM_ATOMIC_LOAD(internal_registry, LIBXSMM_ATOMIC_SEQ_CST);

  if (0 != registry) {
    int i;
#if !defined(LIBXSMM_OPENMP)
    /* acquire locks and thereby shortcut lazy initialization later on */
    const int nlocks = sizeof(internal_reglock) / sizeof(*internal_reglock);
    for (i = 0; i < nlocks; ++i) LIBXSMM_LOCK_ACQUIRE(internal_reglock[i]);
#else
#   pragma omp critical(internal_reglock)
#endif
    {
      registry = internal_registry;

      if (0 != registry) {
        internal_regkey *const registry_keys = internal_registry_keys;
        const char *const target_arch = internal_get_target_arch(internal_target_archid);
        unsigned int heapmem = (LIBXSMM_REGSIZE) * (sizeof(internal_regentry) + sizeof(internal_regkey));

        /* serves as an id to invalidate the thread-local cache; never decremented */
        ++internal_teardown;
#if defined(__TRACE)
        i = libxsmm_trace_finalize();
# if !defined(NDEBUG) /* library code is expected to be mute */
        if (EXIT_SUCCESS != i) {
          fprintf(stderr, "LIBXSMM: failed to finalize trace (error #%i)!\n", i);
        }
# endif
#endif
        libxsmm_gemm_finalize();
        libxsmm_gemm_diff_finalize();
        libxsmm_hash_finalize();

        LIBXSMM_ATOMIC_STORE_ZERO(internal_registry, LIBXSMM_ATOMIC_SEQ_CST);
        internal_registry_keys = 0;

        for (i = 0; i < LIBXSMM_REGSIZE; ++i) {
          internal_regentry code = registry[i];
          if (0 != code.function.pmm/*potentially allocated*/) {
            const libxsmm_gemm_descriptor *const desc = &registry_keys[i].descriptor;
            const unsigned long long size = LIBXSMM_MNK_SIZE(desc->m, desc->n, desc->k);
            const int precision = (0 == (LIBXSMM_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1);
            int bucket = 2;
            if (LIBXSMM_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= size) {
              bucket = 0;
            }
            else if (LIBXSMM_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= size) {
              bucket = 1;
            }
            if (0 != code.size/*JIT: actually allocated*/) {
              /* make address valid by clearing an eventual collision flag */
              code.function.imm &= ~LIBXSMM_HASH_COLLISION;
#if defined(LIBXSMM_VTUNE)
              if (0 != code.id && iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
                char jit_code_name[256];
                LIBXSMM_VTUNE_JIT_DESC_TYPE vtune_jit_desc;
                internal_get_code_name(target_arch, desc,
                  sizeof(jit_code_name), jit_code_name);
                internal_get_vtune_jitdesc(&code, jit_code_name, &vtune_jit_desc);
                iJIT_NotifyEvent(LIBXSMM_VTUNE_JIT_UNLOAD, &vtune_jit_desc);
              }
#endif
#if defined(_WIN32)
              /* TODO: executable memory buffer under Windows */
#else
# if defined(NDEBUG)
              munmap(code.function.pmm, code.size);
# else /* library code is expected to be mute */
              if (0 != munmap(code.function.pmm, code.size)) {
                const int error = errno;
                fprintf(stderr, "LIBXSMM: %s (munmap error #%i at %p+%u)!\n",
                  strerror(error), error, code.function.pmm, code.size);
              }
# endif
#endif
              ++internal_statistic[precision][bucket].njit;
              heapmem += code.size;
            }
            else {
              ++internal_statistic[precision][bucket].nsta;
            }
          }
        }
        if (0 != internal_verbose) { /* print statistic on termination */
          LIBXSMM_FLOCK(stderr);
          LIBXSMM_FLOCK(stdout);
          fflush(stdout); /* synchronize with standard output */
          {
            const unsigned int linebreak = 0 == internal_print_statistic(stderr, target_arch, 1/*SP*/, 1, 0) ? 1 : 0;
            if (0 == internal_print_statistic(stderr, target_arch, 0/*DP*/, linebreak, 0) && 0 != linebreak) {
              fprintf(stderr, "LIBXSMM_TARGET=%s ", target_arch);
            }
            fprintf(stderr, "HEAP: %.f MB\n", 1.0 * heapmem / (1 << 20));
          }
          LIBXSMM_FUNLOCK(stdout);
          LIBXSMM_FUNLOCK(stderr);
        }
        free((void*)registry_keys);
        free((void*)registry);
      }
    }
#if !defined(LIBXSMM_OPENMP) /* release locks */
  for (i = 0; i < nlocks; ++i) LIBXSMM_LOCK_RELEASE(internal_reglock[i]);
#endif
  }
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_get_target_archid(void)
{
  LIBXSMM_INIT
#if !defined(_WIN32) && !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  return internal_target_archid;
#else /* no JIT support */
  return LIBXSMM_MIN(internal_target_archid, LIBXSMM_X86_SSE4_2);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_set_target_archid(int id)
{
  switch (id) {
    case LIBXSMM_X86_AVX512_CORE:
    case LIBXSMM_X86_AVX512_MIC:
    case LIBXSMM_X86_AVX2:
    case LIBXSMM_X86_AVX:
    case LIBXSMM_X86_SSE4_2:
    case LIBXSMM_X86_SSE4_1:
    case LIBXSMM_X86_SSE3:
    case LIBXSMM_TARGET_ARCH_GENERIC: {
      internal_target_archid = id;
    } break;
    default: if (LIBXSMM_X86_GENERIC <= id) {
      internal_target_archid = LIBXSMM_X86_GENERIC;
    }
    else {
      internal_target_archid = LIBXSMM_TARGET_ARCH_UNKNOWN;
    }
  }

#if !defined(NDEBUG) /* library code is expected to be mute */
  {
    const int cpuid = libxsmm_cpuid_x86();
    if (cpuid < internal_target_archid) {
      fprintf(stderr, "LIBXSMM: \"%s\" code will fail to run on \"%s\"!\n",
        internal_get_target_arch(internal_target_archid),
        internal_get_target_arch(cpuid));
    }
  }
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE const char* libxsmm_get_target_arch(void)
{
  LIBXSMM_INIT
  return internal_get_target_arch(internal_target_archid);
}


/* function serves as a helper for implementing the Fortran interface */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void get_target_arch(char* arch, int length);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void get_target_arch(char* arch, int length)
{
  const char* c = libxsmm_get_target_arch();
  int i;
  assert(0 != arch); /* valid here since function is not in the public interface */
  for (i = 0; i < length && 0 != *c; ++i, ++c) arch[i] = *c;
  for (; i < length; ++i) arch[i] = ' ';
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_set_target_arch(const char* arch)
{
  int target_archid = LIBXSMM_TARGET_ARCH_UNKNOWN;

  if (arch && *arch) {
    const int jit = atoi(arch);
    if (0 == strcmp("0", arch)) {
      target_archid = LIBXSMM_TARGET_ARCH_GENERIC;
    }
    else if (1 < jit) {
      target_archid = LIBXSMM_X86_GENERIC + jit;
    }
    else if (0 == strcmp("skx", arch) || 0 == strcmp("avx3", arch) || 0 == strcmp("avx512", arch)) {
      target_archid = LIBXSMM_X86_AVX512_CORE;
    }
    else if (0 == strcmp("knl", arch) || 0 == strcmp("mic2", arch)) {
      target_archid = LIBXSMM_X86_AVX512_MIC;
    }
    else if (0 == strcmp("hsw", arch) || 0 == strcmp("avx2", arch)) {
      target_archid = LIBXSMM_X86_AVX2;
    }
    else if (0 == strcmp("snb", arch) || 0 == strcmp("avx", arch)) {
      target_archid = LIBXSMM_X86_AVX;
    }
    else if (0 == strcmp("wsm", arch) || 0 == strcmp("nhm", arch) || 0 == strcmp("sse4", arch) || 0 == strcmp("sse4_2", arch) || 0 == strcmp("sse4.2", arch)) {
      target_archid = LIBXSMM_X86_SSE4_2;
    }
    else if (0 == strcmp("sse4_1", arch) || 0 == strcmp("sse4.1", arch)) {
      target_archid = LIBXSMM_X86_SSE4_1;
    }
    else if (0 == strcmp("sse3", arch) || 0 == strcmp("sse", arch)) {
      target_archid = LIBXSMM_X86_SSE3;
    }
    else if (0 == strcmp("x86", arch) || 0 == strcmp("sse2", arch)) {
      target_archid = LIBXSMM_X86_GENERIC;
    }
    else if (0 == strcmp("generic", arch) || 0 == strcmp("none", arch)) {
      target_archid = LIBXSMM_TARGET_ARCH_GENERIC;
    }
  }

  if (LIBXSMM_TARGET_ARCH_UNKNOWN == target_archid || LIBXSMM_X86_AVX512_CORE < target_archid) {
    target_archid = libxsmm_cpuid_x86();
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    const int cpuid = libxsmm_cpuid_x86();
    if (cpuid < target_archid) {
      fprintf(stderr, "LIBXSMM: \"%s\" code will fail to run on \"%s\"!\n",
        internal_get_target_arch(target_archid),
        internal_get_target_arch(cpuid));
    }
  }
#endif
  internal_target_archid = target_archid;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_build(const libxsmm_gemm_descriptor* desc, internal_regentry* code)
{
#if (0 != LIBXSMM_JIT)
# if !defined(_WIN32) && !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  const char *const target_arch = internal_get_target_arch(internal_target_archid);
  libxsmm_generated_code generated_code;
  assert(0 != desc && 0 != code);
  assert(0 != internal_target_archid);
  assert(0 == code->function.pmm);

  /* allocate temporary buffer which is large enough to cover the generated code */
  generated_code.generated_code = malloc(131072);
  generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
  generated_code.code_size = 0;
  generated_code.code_type = 2;
  generated_code.last_error = 0;

  /* generate kernel */
  libxsmm_generator_gemm_kernel(&generated_code, desc, target_arch);

  /* handle an eventual error in the else-branch */
  if (0 == generated_code.last_error) {
# if defined(__APPLE__) && defined(__MACH__)
    const int fd = 0;
# else
    const int fd = open("/dev/zero", O_RDWR);
# endif
    if (0 <= fd) {
      /* create executable buffer */
      code->function.pmm = mmap(0, generated_code.code_size,
        /* must be a superset of what mprotect populates (see below) */
        PROT_READ | PROT_WRITE | PROT_EXEC,
# if defined(__APPLE__) && defined(__MACH__)
        LIBXSMM_INTERNAL_MAP | MAP_ANON, fd, 0);
# elif !defined(__CYGWIN__)
        LIBXSMM_INTERNAL_MAP | MAP_32BIT, fd, 0);
      close(fd);
# else
        LIBXSMM_INTERNAL_MAP, fd, 0);
      close(fd);
# endif
      if (MAP_FAILED != code->function.pmm) {
        /* explicitly disable THP for this memory region, kernel 2.6.38 or higher */
# if defined(MADV_NOHUGEPAGE)
#  if defined(NDEBUG)
        madvise(code->function.pmm, generated_code.code_size, MADV_NOHUGEPAGE);
#  else /* library code is expected to be mute */
        /* proceed even in case of an error, we then just take what we got (THP) */
        if (0 != madvise(code->function.pmm, generated_code.code_size, MADV_NOHUGEPAGE)) {
          static LIBXSMM_TLS int once = 0;
          if (0 == once) {
            const int error = errno;
            fprintf(stderr, "LIBXSMM: %s (madvise error #%i at %p)!\n",
              strerror(error), error, code->function.pmm);
            once = 1;
          }
        }
#  endif /*defined(NDEBUG)*/
# elif !(defined(__APPLE__) && defined(__MACH__)) && !defined(__CYGWIN__)
        LIBXSMM_MESSAGE("================================================================================")
        LIBXSMM_MESSAGE("LIBXSMM: Adjusting THP is unavailable due to C89 or kernel older than 2.6.38!")
        LIBXSMM_MESSAGE("================================================================================")
# endif /*MADV_NOHUGEPAGE*/
        /* copy temporary buffer into the prepared executable buffer */
        memcpy(code->function.pmm, generated_code.generated_code, generated_code.code_size);

        if (0/*ok*/ == mprotect(code->function.pmm, generated_code.code_size, PROT_EXEC | PROT_READ)) {
# if (!defined(NDEBUG) && defined(_DEBUG)) || defined(LIBXSMM_VTUNE)
          char jit_code_name[256];
          internal_get_code_name(target_arch, desc, sizeof(jit_code_name), jit_code_name);
# endif
          /* finalize code generation */
          code->size = generated_code.code_size;
          /* free temporary/initial code buffer */
          free(generated_code.generated_code);
# if !defined(NDEBUG) && defined(_DEBUG)
          { /* dump byte-code into file */
            FILE *const byte_code = fopen(jit_code_name, "wb");
            if (0 != byte_code) {
              fwrite(code->function.pmm, 1, code->size, byte_code);
              fclose(byte_code);
            }
          }
# endif /*!defined(NDEBUG) && defined(_DEBUG)*/
# if defined(LIBXSMM_VTUNE)
          if (iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
            LIBXSMM_VTUNE_JIT_DESC_TYPE vtune_jit_desc;
            code->id = iJIT_GetNewMethodID();
            internal_get_vtune_jitdesc(code, jit_code_name, &vtune_jit_desc);
            iJIT_NotifyEvent(LIBXSMM_VTUNE_JIT_LOAD, &vtune_jit_desc);
          }
          else {
            code->id = 0;
          }
# endif
        }
        else { /* there was an error with mprotect */
# if defined(NDEBUG)
          munmap(code->function.pmm, generated_code.code_size);
# else /* library code is expected to be mute */
          static LIBXSMM_TLS int once = 0;
          if (0 == once) {
            const int error = errno;
            fprintf(stderr, "LIBXSMM: %s (mprotect error #%i at %p+%u)!\n",
              strerror(error), error, code->function.pmm, generated_code.code_size);
            once = 1;
          }
          if (0 != munmap(code->function.pmm, generated_code.code_size)) {
            static LIBXSMM_TLS int once_mmap_error = 0;
            if (0 == once_mmap_error) {
              const int error = errno;
              fprintf(stderr, "LIBXSMM: %s (munmap error #%i at %p+%u)!\n",
                strerror(error), error, code->function.pmm, generated_code.code_size);
              once_mmap_error = 1;
            }
          }
# endif
          free(generated_code.generated_code);
        }
      }
      else {
# if !defined(NDEBUG) /* library code is expected to be mute */
        static LIBXSMM_TLS int once = 0;
        if (0 == once) {
          const int error = errno;
          fprintf(stderr, "LIBXSMM: %s (mmap allocation error #%i)!\n",
            strerror(error), error);
          once = 1;
        }
# endif
        free(generated_code.generated_code);
        /* clear MAP_FAILED value */
        code->function.pmm = 0;
      }
    }
# if !defined(NDEBUG)/* library code is expected to be mute */
    else {
      static LIBXSMM_TLS int once = 0;
      if (0 == once) {
        fprintf(stderr, "LIBXSMM: invalid file descriptor (%i)\n", fd);
        once = 1;
      }
    }
# endif
  }
  else {
# if !defined(NDEBUG) /* library code is expected to be mute */
    static LIBXSMM_TLS int once = 0;
    if (0 == once) {
      fprintf(stderr, "%s (error #%u)\n", libxsmm_strerror(generated_code.last_error),
        generated_code.last_error);
      once = 1;
    }
# endif
    free(generated_code.generated_code);
  }
# else
#   if !defined(__MIC__)
  LIBXSMM_MESSAGE("================================================================================")
  LIBXSMM_MESSAGE("LIBXSMM: The JIT BACKEND is currently not supported under Microsoft Windows!")
  LIBXSMM_MESSAGE("================================================================================")
#   endif
  LIBXSMM_UNUSED(desc); LIBXSMM_UNUSED(code);
  /* libxsmm_get_target_arch also serves as a runtime check whether JIT is available or not */
  assert(LIBXSMM_X86_AVX > internal_target_archid);
# endif /*_WIN32*/
#endif /*LIBXSMM_JIT*/
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE libxsmm_xmmfunction internal_xmmdispatch(const libxsmm_gemm_descriptor* descriptor)
{
  INTERNAL_FIND_CODE_DECLARE(code);
  assert(descriptor);
  {
    INTERNAL_FIND_CODE(descriptor, code);
  }
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_xmmfunction libxsmm_xmmdispatch(const libxsmm_gemm_descriptor* descriptor)
{
  const libxsmm_xmmfunction null_mmfunction = { 0 };
  return 0 != descriptor ? internal_xmmdispatch(descriptor) : null_mmfunction;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_smmfunction libxsmm_smmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch)
{
  INTERNAL_SMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dmmfunction libxsmm_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
  INTERNAL_DMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


#if defined(LIBXSMM_GEMM_EXTWRAP)
#if defined(__STATIC)

LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(__real_sgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  sgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(__real_dgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  dgemm_(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*defined(__STATIC)*/
#endif /*defined(LIBXSMM_GEMM_EXTWRAP)*/

