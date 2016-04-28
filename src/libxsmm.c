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
#include "libxsmm_hash.h"
#include "libxsmm_gemm.h"

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
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif
#if defined(__GNUC__)
# if !defined(LIBXSMM_GCCATOMICS)
#   if (LIBXSMM_VERSION3(4, 7, 4) <= LIBXSMM_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#     define LIBXSMM_GCCATOMICS 1
#   else
#     define LIBXSMM_GCCATOMICS 0
#   endif
# endif
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
#   define LIBXSMM_HASH_BASIC
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
# define LIBXSMM_HASH_FUNCTION libxsmm_hash_npot
# define LIBXSMM_HASH_FUNCTION_CALL(HASH, INDX, HASH_FUNCTION, DESCRIPTOR) \
    HASH = (HASH_FUNCTION)(&(DESCRIPTOR), LIBXSMM_GEMM_DESCRIPTOR_SIZE, LIBXSMM_REGSIZE); \
    assert((LIBXSMM_REGSIZE) > (HASH)); \
    INDX = (HASH)
#else
# define LIBXSMM_HASH_FUNCTION libxsmm_crc32
# define LIBXSMM_HASH_FUNCTION_CALL(HASH, INDX, HASH_FUNCTION, DESCRIPTOR) \
    HASH = (HASH_FUNCTION)(&(DESCRIPTOR), LIBXSMM_GEMM_DESCRIPTOR_SIZE, 25071975/*seed*/); \
    INDX = LIBXSMM_HASH_MOD(HASH, LIBXSMM_REGSIZE)
#endif

/* flag fused into the memory address of a code version in case of collision */
#define LIBXSMM_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 1))

typedef union LIBXSMM_RETARGETABLE internal_regkey {
  char simd[32];
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
} internal_regentry;

LIBXSMM_DEBUG(LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL unsigned int internal_ncollisions = 0;)
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL internal_regkey* internal_registry_keys = 0;
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL internal_regentry* internal_registry = 0;

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
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL int internal_target_arch = LIBXSMM_TARGET_ARCH_GENERIC;
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL const char* internal_target_archid = 0;

#if !defined(LIBXSMM_OPENMP)
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL LIBXSMM_LOCK_TYPE internal_reglock[] = {
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT,
  LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT, LIBXSMM_LOCK_CONSTRUCT
};
#endif

#if defined(__GNUC__)
  /* libxsmm_init already executed via GCC constructor attribute */
# define INTERNAL_FIND_CODE_INIT(VARIABLE) assert(0 != (VARIABLE))
#else /* lazy initialization */
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

#if (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
#   define INTERNAL_FIND_CODE_DECLARE(CODE) internal_regentry* CODE = __atomic_load_n(&internal_registry, __ATOMIC_RELAXED); unsigned int i
#   define INTERNAL_FIND_CODE_READ(CODE, DST) DST = __atomic_load_n(&((CODE)->function.pmm), __ATOMIC_SEQ_CST)
#   define INTERNAL_FIND_CODE_WRITE(CODE, SRC) __atomic_store_n(&(CODE)->function.pmm, SRC, __ATOMIC_SEQ_CST);
# else
#   define INTERNAL_FIND_CODE_DECLARE(CODE) internal_regentry* CODE = __sync_or_and_fetch(&internal_registry, 0); unsigned int i
#   define INTERNAL_FIND_CODE_READ(CODE, DST) DST = __sync_or_and_fetch(&((CODE)->function.pmm), 0)
#   define INTERNAL_FIND_CODE_WRITE(CODE, SRC) { \
      /*const*/void* old = (CODE)->function.pmm; \
      while (!__sync_bool_compare_and_swap(&(CODE)->function.pmm, old, SRC)) { \
        old = (CODE)->function.pmm; \
      } \
    }
# endif
#elif (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(_WIN32) /*TODO*/
# define INTERNAL_FIND_CODE_DECLARE(CODE) internal_regentry* CODE = internal_registry; unsigned int i
# define INTERNAL_FIND_CODE_READ(CODE, DST) DST = (CODE)->function.pmm
# define INTERNAL_FIND_CODE_WRITE(CODE, SRC) (CODE)->function.pmm = (SRC)
#else
# define INTERNAL_FIND_CODE_DECLARE(CODE) internal_regentry* CODE = internal_registry; unsigned int i
# define INTERNAL_FIND_CODE_READ(CODE, DST) DST = (CODE)->function.pmm
# define INTERNAL_FIND_CODE_WRITE(CODE, SRC) (CODE)->function.pmm = (SRC)
#endif

#if defined(LIBXSMM_CACHESIZE) && (0 < LIBXSMM_CACHESIZE)
# define INTERNAL_FIND_CODE_CACHE_DECL(CACHE_KEYS, CACHE) \
  static LIBXSMM_TLS union { char padding[32]; libxsmm_gemm_descriptor desc; } CACHE_KEYS[LIBXSMM_CACHESIZE]; \
  static LIBXSMM_TLS libxsmm_xmmfunction CACHE[LIBXSMM_CACHESIZE]; \
  static LIBXSMM_TLS unsigned int cache_hit = LIBXSMM_CACHESIZE
# define INTERNAL_FIND_CODE_CACHE_BEGIN(DESCRIPTOR, CACHE_KEYS, CACHE, RESULT) \
  assert(32 >= LIBXSMM_GEMM_DESCRIPTOR_SIZE); \
  /* search small cache starting with the last hit on record */ \
  i = libxsmm_gemm_diffn(DESCRIPTOR, &((CACHE_KEYS)->desc), cache_hit, LIBXSMM_CACHESIZE, 32); \
  if (LIBXSMM_CACHESIZE > i) { /* cache hit */ \
    (RESULT).function.xmm = (CACHE)[i]; \
    cache_hit = i; \
  } \
  else
# if defined(LIBXSMM_GEMM_DIFF_SW) && (2 == (LIBXSMM_GEMM_DIFF_SW)) /* most general implementation */
#   define INTERNAL_FIND_CODE_CACHE_FINALIZE(DESCRIPTOR, CACHE_KEYS, CACHE, RESULT) \
    i = (cache_hit + LIBXSMM_CACHESIZE - 1) % LIBXSMM_CACHESIZE; \
    ((CACHE_KEYS)[i]).desc = *(DESCRIPTOR); \
    (CACHE)[i] = (RESULT).function.xmm; \
    cache_hit = i
# else
#   define INTERNAL_FIND_CODE_CACHE_FINALIZE(DESCRIPTOR, CACHE_KEYS, CACHE, RESULT) \
    assert(/*is pot*/LIBXSMM_CACHESIZE == (1 << LIBXSMM_LOG2(LIBXSMM_CACHESIZE))); \
    i = LIBXSMM_MOD2(cache_hit + LIBXSMM_CACHESIZE - 1, LIBXSMM_CACHESIZE); \
    (CACHE_KEYS)[i].desc = *(DESCRIPTOR); \
    (CACHE)[i] = (RESULT).function.xmm; \
    cache_hit = i
# endif
#else
# define INTERNAL_FIND_CODE_CACHE_DECL(CACHE_KEYS, CACHE)
# define INTERNAL_FIND_CODE_CACHE_BEGIN(DESCRIPTOR, CACHE_KEYS, CACHE, RESULT)
# define INTERNAL_FIND_CODE_CACHE_FINALIZE(DESCRIPTOR, CACHE_KEYS, CACHE, RESULT)
#endif

#if (0 != LIBXSMM_JIT)
# define INTERNAL_FIND_CODE_JIT(DESCRIPTOR, CODE, RESULT) \
  /* check if code generation or fix-up is needed, also check whether JIT is supported (CPUID) */ \
  if (0 == (RESULT).function.pmm && LIBXSMM_X86_AVX <= internal_target_arch) { \
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
          LIBXSMM_DEBUG(++internal_ncollisions;) \
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

#define INTERNAL_FIND_CODE(DESCRIPTOR, CODE, HASH_FUNCTION, DIFF_FUNCTION) \
  internal_regentry flux_entry; \
{ \
  INTERNAL_FIND_CODE_CACHE_DECL(cache_keys, cache); \
  unsigned int hash, diff = 0, diff0 = 0, i0; \
  INTERNAL_FIND_CODE_INIT(CODE); \
  INTERNAL_FIND_CODE_CACHE_BEGIN(DESCRIPTOR, cache_keys, cache, flux_entry) { \
    /* check if the requested xGEMM is already JITted */ \
    LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */ \
    LIBXSMM_HASH_FUNCTION_CALL(hash, i = i0, HASH_FUNCTION, *(DESCRIPTOR)); \
    (CODE) += i; /* actual entry */ \
    do { \
      INTERNAL_FIND_CODE_READ(CODE, flux_entry.function.pmm); /* read registered code */ \
      if (0 != flux_entry.function.pmm) { \
        if (0 == diff0) { \
          if (0 == (LIBXSMM_HASH_COLLISION & flux_entry.function.imm)) { /* check for no collision */ \
            /* calculate bitwise difference (deep check) */ \
            LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */ \
            diff = (DIFF_FUNCTION)(DESCRIPTOR, &internal_registry_keys[i].descriptor); \
            if (0 != diff) { /* new collision discovered (but no code version yet) */ \
              /* allow to fix-up current entry inside of the guarded/locked region */ \
              flux_entry.function.pmm = 0; \
            } \
          } \
          /* collision discovered but code version exists; perform deep check */ \
          else if (0 != (DIFF_FUNCTION)(DESCRIPTOR, &internal_registry_keys[i].descriptor)) { \
            /* continue linearly searching code starting at re-hashed index position */ \
            const unsigned int index = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_REGSIZE); \
            unsigned int next; \
            for (i0 = (index != i ? index : LIBXSMM_HASH_MOD(index + 1, LIBXSMM_REGSIZE)), \
              i = i0, next = LIBXSMM_HASH_MOD(i0 + 1, LIBXSMM_REGSIZE); \
              /* skip any (still invalid) descriptor which corresponds to no code, or continue on difference */ \
              (0 == (CODE = (internal_registry + i))->function.pmm || \
                0 != (diff = (DIFF_FUNCTION)(DESCRIPTOR, &internal_registry_keys[i].descriptor))) \
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
    INTERNAL_FIND_CODE_CACHE_FINALIZE(DESCRIPTOR, cache_keys, cache, flux_entry); \
  } \
} \
return flux_entry.function.xmm

#define INTERNAL_DISPATCH_MAIN(DESCRIPTOR_DECL, DESC, FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION) { \
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
      INTERNAL_FIND_CODE(DESC, code, HASH_FUNCTION, DIFF_FUNCTION).SELECTOR; \
    } \
  } \
  else { /* TODO: not supported (bypass) */ \
    return 0; \
  } \
}

#if defined(LIBXSMM_GEMM_DIFF_MASK_A) /* no padding i.e., LIBXSMM_GEMM_DESCRIPTOR_SIZE */
# define INTERNAL_DISPATCH(FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION) \
    INTERNAL_DISPATCH_MAIN(libxsmm_gemm_descriptor descriptor, &descriptor, \
    FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION)
#else /* padding: LIBXSMM_GEMM_DESCRIPTOR_SIZE -> LIBXSMM_ALIGNMENT */
# define INTERNAL_DISPATCH(FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION) { \
    INTERNAL_DISPATCH_MAIN(union { libxsmm_gemm_descriptor desc; char simd[LIBXSMM_ALIGNMENT]; } simd_descriptor; \
      for (i = LIBXSMM_GEMM_DESCRIPTOR_SIZE; i < sizeof(simd_descriptor.simd); ++i) simd_descriptor.simd[i] = 0, &simd_descriptor.desc, \
    FLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, SELECTOR/*smm or dmm*/, HASH_FUNCTION, DIFF_FUNCTION)
#endif

#define INTERNAL_SMMDISPATCH(PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, HASH_FUNCTION, DIFF_FUNCTION) \
  INTERNAL_DISPATCH((0 == (PFLAGS) ? LIBXSMM_FLAGS : *(PFLAGS)) | LIBXSMM_GEMM_FLAG_F32PREC, \
  M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, smm, HASH_FUNCTION, DIFF_FUNCTION)

#define INTERNAL_DMMDISPATCH(PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, HASH_FUNCTION, DIFF_FUNCTION) \
  INTERNAL_DISPATCH((0 == (PFLAGS) ? LIBXSMM_FLAGS : *(PFLAGS)), \
  M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH, dmm, HASH_FUNCTION, DIFF_FUNCTION)


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_register_static_code(
  const libxsmm_gemm_descriptor* desc, unsigned int index, unsigned int hash, libxsmm_xmmfunction src,
  internal_regentry* dst_entries, unsigned int* registered, unsigned int* total)
{
  internal_regkey* dst_keys = internal_registry_keys;
  assert(0 != desc && 0 != src.dmm && 0 != dst_keys && 0 != dst_entries && 0 != registered && 0 != total);

  if (0 != dst_entries->function.pmm) { /* collision? */
    /* start at a re-hashed index position */
    const unsigned int start = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_REGSIZE);
    internal_regentry *const registry = dst_entries - index; /* recalculate base address */
    unsigned int i0, i, next;

    /* mark current entry as a collision (this might be already the case) */
    dst_entries->function.imm |= LIBXSMM_HASH_COLLISION;

    /* start linearly searching for an available slot */
    for (i = (start != index) ? start : LIBXSMM_HASH_MOD(start + 1, LIBXSMM_REGSIZE), i0 = i, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE);
      0 != (dst_entries = registry + i)->function.pmm && next != i0; i = next, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE));

    dst_keys += i;
  }

  if (0 == dst_entries->function.pmm) { /* registry not (yet) exhausted */
    dst_entries->function.xmm = src;
    dst_entries->size = 0; /* statically generated code */
    dst_keys->descriptor = *desc;
    ++(*registered);
  }

  ++(*total);
}


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
#if (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
    result = __atomic_load_n(&internal_registry, __ATOMIC_SEQ_CST);
# else
    result = __sync_or_and_fetch(&internal_registry, 0);
# endif
#elif (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(_WIN32)
    result = internal_registry; /*TODO*/
#else
    result = internal_registry;
#endif
    if (0 == result) {
      int init_code;
      const char *const env_jit = getenv("LIBXSMM_JIT");
      if (env_jit && *env_jit) {
        const int jit = atoi(env_jit);
        if (0 == strcmp("0", env_jit)) { /* suppress running libxsmm_cpuid_x86 */
          internal_target_archid = "generic";
        }
        else if (1 < jit) { /* suppress libxsmm_cpuid_x86 and override archid */
          switch (LIBXSMM_X86_GENERIC + jit) {
            case LIBXSMM_X86_AVX512: {
              internal_target_arch = LIBXSMM_X86_AVX512;
              internal_target_archid = "knl"; /* "skx" is fine too */
            } break;
            case LIBXSMM_X86_AVX2: {
              internal_target_arch = LIBXSMM_X86_AVX2;
              internal_target_archid = "hsw";
            } break;
            case LIBXSMM_X86_AVX: {
              internal_target_arch = LIBXSMM_X86_AVX;
              internal_target_archid = "snb";
            } break;
            default: if (LIBXSMM_X86_SSE3 <= (LIBXSMM_X86_GENERIC + jit)) {
              internal_target_arch = LIBXSMM_X86_GENERIC + jit;
              internal_target_archid = "sse";
            }
          }
        }
        else if (0 == strcmp("knl", env_jit) || 0 == strcmp("skx", env_jit)) {
          internal_target_arch = LIBXSMM_X86_AVX512;
          internal_target_archid = env_jit;
        }
        else if (0 == strcmp("hsw", env_jit)) {
          internal_target_arch = LIBXSMM_X86_AVX2;
          internal_target_archid = env_jit;
        }
        else if (0 == strcmp("snb", env_jit)) {
          internal_target_arch = LIBXSMM_X86_AVX;
          internal_target_archid = env_jit;
        }
      }
      if (0 == internal_target_archid) {
        internal_target_arch = libxsmm_cpuid_x86(&internal_target_archid);
        assert(0 != internal_target_archid);
      }
      { /* select prefetch strategy for JIT */
        const char *const env_prefetch = getenv("LIBXSMM_PREFETCH");
        if (0 == env_prefetch || 0 == *env_prefetch) {
          if (0 > LIBXSMM_PREFETCH) { /* permitted by LIBXSMM_PREFETCH_AUTO */
            assert(0 != internal_target_archid);
            internal_prefetch = 0 != strcmp("knl", internal_target_archid)
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
      libxsmm_hash_init(internal_target_arch);
      libxsmm_gemm_diff_init(internal_target_arch);
      init_code = libxsmm_gemm_init(internal_target_archid, internal_prefetch);
#if defined(__TRACE)
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
#endif
      if (EXIT_SUCCESS == init_code) {
        assert(0 == internal_registry_keys/*should never happen*/);
        result = (internal_regentry*)malloc(LIBXSMM_REGSIZE * sizeof(internal_regentry));
        internal_registry_keys = (internal_regkey*)malloc(LIBXSMM_REGSIZE * sizeof(internal_regkey));

        if (result && internal_registry_keys) {
          for (i = 0; i < LIBXSMM_REGSIZE; ++i) result[i].function.pmm = 0;
          /* omit registering code if JIT is enabled and if an ISA extension is found
           * which is beyond the static code path used to compile the library
           */
#if (0 != LIBXSMM_JIT) && !defined(__MIC__)
          if (LIBXSMM_STATIC_TARGET_ARCH >= internal_target_arch)
#endif
          { /* opening a scope for eventually declaring variables */
            unsigned int csp_tot = 0, csp_reg = 0, cdp_tot = 0, cdp_reg = 0;
            /* setup the dispatch table for the statically generated code */
#           include <libxsmm_dispatch.h>
#if !defined(NDEBUG) /* library code is expected to be mute */
            if (csp_reg < csp_tot) {
              fprintf(stderr, "LIBXSMM: %u of %u SP-kernels are not registered due to hash key collisions!\n", csp_tot - csp_reg, csp_tot);
            }
            if (cdp_reg < cdp_tot) {
              fprintf(stderr, "LIBXSMM: %u of %u DP-kernels are not registered due to hash key collisions!\n", cdp_tot - cdp_reg, cdp_tot);
            }
#else
            LIBXSMM_UNUSED(csp_tot); LIBXSMM_UNUSED(csp_reg); LIBXSMM_UNUSED(cdp_tot); LIBXSMM_UNUSED(cdp_reg);
#endif
          }
          atexit(libxsmm_finalize);
#if (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
          __atomic_store_n(&internal_registry, result, __ATOMIC_SEQ_CST);
# else
          {
            internal_regentry* old = internal_registry;
            while (!__sync_bool_compare_and_swap(&internal_registry, old, result)) old = internal_registry;
          }
# endif
#elif (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(_WIN32)
          internal_registry = result; /*TODO*/
#else
          internal_registry = result;
#endif
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
#if (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
  const void *const registry = __atomic_load_n(&internal_registry, __ATOMIC_RELAXED);
# else
  const void *const registry = __sync_or_and_fetch(&internal_registry, 0);
# endif
#elif (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(_WIN32)
  const void *const registry = internal_registry; /*TODO*/
#else
  const void *const registry = internal_registry;
#endif
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
#if (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
  internal_regentry* registry = __atomic_load_n(&internal_registry, __ATOMIC_SEQ_CST);
# else
  internal_regentry* registry = __sync_or_and_fetch(&internal_registry, 0);
# endif
#elif (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(_WIN32)
  internal_regentry* registry = internal_registry; /*TODO*/
#else
  internal_regentry* registry = internal_registry;
#endif

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
        void *const registry_keys = internal_registry_keys;
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
#if (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(LIBXSMM_GCCATOMICS)
# if (0 != LIBXSMM_GCCATOMICS)
        __atomic_store_n(&internal_registry, 0, __ATOMIC_SEQ_CST);
# else
        { /* use store side-effect of built-in (dummy assignment to mute warning) */
          internal_regentry *const dummy = __sync_and_and_fetch(&internal_registry, 0);
          LIBXSMM_UNUSED(dummy);
        }
# endif
#elif (defined(_REENTRANT) || defined(LIBXSMM_OPENMP)) && defined(_WIN32)
        internal_registry = 0; /*TODO*/
#else
        internal_registry = 0;
#endif
        internal_registry_keys = 0;
        { /* open scope to allocate variables */
          LIBXSMM_DEBUG(unsigned int njit = 0, nstatic = 0;)
          for (i = 0; i < LIBXSMM_REGSIZE; ++i) {
            internal_regentry code = registry[i];
            if (0 != code.function.pmm/*potentially allocated*/) {
              if (0 != code.size/*JIT: actually allocated*/) {
                /* make address valid by clearing an eventual collision flag */
                code.function.imm &= ~LIBXSMM_HASH_COLLISION;
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
                LIBXSMM_DEBUG(++njit;)
              }
              else {
                LIBXSMM_DEBUG(++nstatic;)
              }
            }
          }
#if !defined(NDEBUG) /* library code is expected to be mute */
          fprintf(stderr, "LIBXSMM_JIT=%s NJIT=%u NSTATIC=%u", 0 != internal_target_archid ? internal_target_archid : "0", njit, nstatic);
          if (0 != internal_ncollisions) {
            fprintf(stderr, ": %u hash key collisions handled!\n", internal_ncollisions);
          }
          else {
            fprintf(stderr, "\n");
          }
#endif
        }
        free((void*)registry);
        free(registry_keys);
      }
    }
#if !defined(LIBXSMM_OPENMP) /* release locks */
  for (i = 0; i < nlocks; ++i) LIBXSMM_LOCK_RELEASE(internal_reglock[i]);
#endif
  }
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_get_target_arch()
{
#if !defined(_WIN32) && !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  return internal_target_arch;
#else /* no JIT support */
  return LIBXSMM_TARGET_ARCH_GENERIC;
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE const char* libxsmm_get_target_archid()
{
  return internal_target_archid;
}


/* function serves as a helper for implementing the Fortran interface */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void get_target_archid(char* name, int length)
{
  const char* c = internal_target_archid ? internal_target_archid : "";
  int i;
  assert(0 != name); /* valid here since function is not in the public interface */
  for (i = 0; i < length && 0 != *c; ++i, ++c) name[i] = *c;
  for (; i < length; ++i) name[i] = ' ';
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_build(const libxsmm_gemm_descriptor* desc, internal_regentry* code)
{
#if (0 != LIBXSMM_JIT)
# if !defined(_WIN32) && !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
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
  libxsmm_generator_gemm_kernel(&generated_code, desc, internal_target_archid);

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
# if !defined(NDEBUG) && defined(_DEBUG)
          /* write buffer for manual decode as binary to a file */
          char objdump_name[512];
          FILE* byte_code;
          sprintf(objdump_name, "kernel_%s_f%i_%c%c_m%u_n%u_k%u_lda%u_ldb%u_ldc%u_a%i_b%i_pf%i.bin",
            internal_target_archid /* best available/supported code path */,
            0 == (LIBXSMM_GEMM_FLAG_F32PREC & desc->flags) ? 64 : 32,
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & desc->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & desc->flags) ? 'n' : 't',
            desc->m, desc->n, desc->k, desc->lda, desc->ldb, desc->ldc,
            desc->alpha, desc->beta, desc->prefetch);
          byte_code = fopen(objdump_name, "wb");
          if (0 != byte_code) {
            fwrite(generated_code.generated_code, 1, generated_code.code_size, byte_code);
            fclose(byte_code);
          }
# endif /*!defined(NDEBUG) && defined(_DEBUG)*/
          /* free temporary/initial code buffer */
          free(generated_code.generated_code);
          /* finalize code generation */
          code->size = generated_code.code_size;
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
  assert(LIBXSMM_X86_AVX > libxsmm_get_target_arch());
# endif /*_WIN32*/
#endif /*LIBXSMM_JIT*/
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE libxsmm_xmmfunction internal_xmmdispatch(const libxsmm_gemm_descriptor* descriptor)
{
  INTERNAL_FIND_CODE_DECLARE(code);
  assert(descriptor);
  {
#if defined(LIBXSMM_HASH_BASIC)
    INTERNAL_FIND_CODE(descriptor, code, libxsmm_hash_npot, libxsmm_gemm_diff);
#else
    INTERNAL_FIND_CODE(descriptor, code, libxsmm_crc32, libxsmm_gemm_diff);
#endif
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
#if defined(LIBXSMM_HASH_BASIC)
  INTERNAL_SMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxsmm_hash_npot, libxsmm_gemm_diff);
#else
  INTERNAL_SMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxsmm_crc32, libxsmm_gemm_diff);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dmmfunction libxsmm_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
#if defined(LIBXSMM_HASH_BASIC)
  INTERNAL_DMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxsmm_hash_npot, libxsmm_gemm_diff);
#else
  INTERNAL_DMMDISPATCH(flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch, libxsmm_crc32, libxsmm_gemm_diff);
#endif
}

