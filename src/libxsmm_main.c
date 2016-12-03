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
#include "libxsmm_gemm_diff.h"
#include "libxsmm_trans.h"
#include "libxsmm_gemm.h"
#include "libxsmm_hash.h"
#include "libxsmm_main.h"
#if defined(__TRACE)
# include "libxsmm_trace.h"
#endif
#if defined(LIBXSMM_PERF)
# include "libxsmm_perf.h"
#endif
#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
/* mute warning about target attribute; KNC/native plus JIT is disabled below! */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if !defined(NDEBUG)
# include <errno.h>
#endif
#if defined(_WIN32)
# include <Windows.h>
#else
# include <sys/mman.h>
# include <unistd.h>
# include <fcntl.h>
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
#if !defined(LIBXSMM_HASH_BASIC)
# if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH) || (LIBXSMM_X86_SSE4_2 > LIBXSMM_MAX_STATIC_TARGET_ARCH)
/*#   define LIBXSMM_HASH_BASIC*/
# endif
#endif

/* LIBXSMM_REGSIZE is POT */
/*#define LIBXSMM_HASH_MOD(N, NGEN) ((N) % (NGEN))*/
#define LIBXSMM_HASH_MOD(N, NPOT) LIBXSMM_MOD2(N, NPOT)

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
/* flag fused into the memory address of a code version in case of non-JIT */
#define LIBXSMM_CODE_STATIC (1ULL << (8 * sizeof(void*) - 2))

#if 16 >= (LIBXSMM_GEMM_DESCRIPTOR_SIZE)
# define LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE 16
#elif 32 >= (LIBXSMM_GEMM_DESCRIPTOR_SIZE)
# define LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE 32
#else
# define LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE LIBXSMM_GEMM_DESCRIPTOR_SIZE
#endif

typedef union LIBXSMM_RETARGETABLE internal_regkey_type {
  char simd[LIBXSMM_GEMM_DESCRIPTOR_SIMD_SIZE];
  libxsmm_gemm_descriptor descriptor;
} internal_regkey_type;

typedef struct LIBXSMM_RETARGETABLE internal_statistic_type {
  unsigned int ntry, ncol, njit, nsta;
} internal_statistic_type;

/** Helper macro determining the default prefetch strategy which is used for statically generated kernels. */
#if (0 > LIBXSMM_PREFETCH) /* auto-prefetch (frontend) */
# if defined(__MIC__) || (LIBXSMM_X86_AVX512_MIC == LIBXSMM_STATIC_TARGET_ARCH)
#   define INTERNAL_PREFETCH LIBXSMM_PREFETCH_AL2BL2_VIA_C
# elif (0 > LIBXSMM_PREFETCH) /* auto-prefetch (frontend) */
#   define INTERNAL_PREFETCH LIBXSMM_PREFETCH_SIGONLY
# endif
#else
# define INTERNAL_PREFETCH LIBXSMM_PREFETCH
#endif

#if !defined(LIBXSMM_TRYLOCK)
/*# define LIBXSMM_TRYLOCK*/
#endif

#if defined(LIBXSMM_CTOR)
/* libxsmm_init already executed via GCC constructor attribute */
# define INTERNAL_FIND_CODE_INIT(VARIABLE) assert(0 != (VARIABLE))
#else /* lazy initialization */
/* use return value of internal_init to refresh local representation */
# define INTERNAL_FIND_CODE_INIT(VARIABLE) if (0 == (VARIABLE)) VARIABLE = internal_init()
#endif

#if defined(LIBXSMM_OPENMP)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX) LIBXSMM_PRAGMA(omp critical(internal_reglock)) { \
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) }
#elif !defined(LIBXSMM_NO_SYNC)
# if defined(LIBXSMM_TRYLOCK)
#   define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX) { \
      const unsigned int LOCKINDEX = LIBXSMM_MOD2(INDEX, INTERNAL_REGLOCK_COUNT); \
      if (LIBXSMM_LOCK_ACQUIRED != LIBXSMM_LOCK_TRYLOCK(internal_reglock + (LOCKINDEX))) { \
        assert(0 == diff); continue; \
      }
# else
#   define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX) { \
      const unsigned int LOCKINDEX = LIBXSMM_MOD2(INDEX, INTERNAL_REGLOCK_COUNT); \
      LIBXSMM_LOCK_ACQUIRE(internal_reglock + (LOCKINDEX))
# endif
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXSMM_LOCK_RELEASE(internal_reglock + (LOCKINDEX)); }
#else
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX)
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX)
#endif

#define INTERNAL_FIND_CODE_DECLARE(CODE) libxsmm_code_pointer* CODE = \
  LIBXSMM_ATOMIC_LOAD(&internal_registry, LIBXSMM_ATOMIC_RELAXED); unsigned int i

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
    (RESULT).xmm = (CACHE)[i]; \
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
    (CACHE)[i] = (RESULT).xmm; \
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
    (CACHE)[i] = (RESULT).xmm; \
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
  if (0 == (RESULT).pmm/*code version does not exist*/ && LIBXSMM_X86_AVX <= libxsmm_target_archid) { \
    /* instead of blocking others, a try-lock allows to let other threads fallback to BLAS during lock-duration */ \
    INTERNAL_FIND_CODE_LOCK(lock, i); /* lock the registry entry */ \
    if (0 == diff) { \
      RESULT = *(CODE); /* deliver code */ \
      /* clear collision flag; can be never static code */ \
      assert(0 == (LIBXSMM_CODE_STATIC & (RESULT).imm)); \
      (RESULT).imm &= ~LIBXSMM_HASH_COLLISION; \
    } \
    if (0 == (RESULT).pmm) { /* double-check (re-read registry entry) after acquiring the lock */ \
      if (0 == diff) { \
        libxsmm_build_request request; \
        request.descriptor.gemm = DESCRIPTOR; request.kind = LIBXSMM_BUILD_KIND_GEMM; \
        /* found a conflict-free registry-slot, and attempt to build the kernel */ \
        libxsmm_build(&request, i, &(RESULT)); \
        internal_update_mmstatistic(DESCRIPTOR, 1, 0); \
        if (0 != (RESULT).pmm) { /* synchronize registry entry */ \
          internal_registry_keys[i].descriptor = *(DESCRIPTOR); \
          *(CODE) = RESULT; \
          LIBXSMM_ATOMIC_STORE(&(CODE)->pmm, (RESULT).pmm, LIBXSMM_ATOMIC_SEQ_CST); \
        } \
      } \
      else { /* 0 != diff */ \
        if (0 == diff0 && /*bypass*/0 == (LIBXSMM_HASH_COLLISION & (CODE)->imm)) { \
          /* flag existing entry as collision */ \
          libxsmm_code_pointer collision; \
          /* find new slot to store the code version */ \
          const unsigned int index = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_REGSIZE); \
          collision.imm = (CODE)->imm | LIBXSMM_HASH_COLLISION; \
          i = (index != i ? index : LIBXSMM_HASH_MOD(index + 1, LIBXSMM_REGSIZE)); \
          i0 = i; /* keep starting point of free-slot-search in mind */ \
          internal_update_mmstatistic(DESCRIPTOR, 0, 1); \
          LIBXSMM_ATOMIC_STORE(&(CODE)->pmm, collision.pmm, LIBXSMM_ATOMIC_SEQ_CST); /* fix-up existing entry */ \
          diff0 = diff; /* no more fix-up */ \
        } \
        else { \
          const unsigned int next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE); \
          if (next != i0) { /* linear search for free slot not completed */ \
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
  libxsmm_code_pointer flux_entry = { 0 }; \
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
      flux_entry.pmm = LIBXSMM_ATOMIC_LOAD(&(CODE)->pmm, LIBXSMM_ATOMIC_SEQ_CST); /* read registered code */ \
      if (0 != flux_entry.pmm) { /* code version exists */ \
        if (0 == diff0) { \
          if (0 == (LIBXSMM_HASH_COLLISION & flux_entry.imm)) { /* check for no collision */ \
            /* calculate bitwise difference (deep check) */ \
            LIBXSMM_PRAGMA_FORCEINLINE /* must precede a statement */ \
            diff = libxsmm_gemm_diff(DESCRIPTOR, &internal_registry_keys[i].descriptor); \
            if (0 != diff) { /* new collision discovered (but no code version yet) */ \
              /* allow to fix-up current entry inside of the guarded/locked region */ \
              flux_entry.pmm = 0; \
            } \
            else { \
              flux_entry.imm &= ~LIBXSMM_CODE_STATIC; \
            } \
          } \
          /* collision discovered but code version exists; perform deep check */ \
          else if (0 != libxsmm_gemm_diff(DESCRIPTOR, &internal_registry_keys[i].descriptor)) { \
            /* continue linearly searching code starting at re-hashed index position */ \
            const unsigned int index = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_REGSIZE); \
            unsigned int next; \
            for (i0 = (index != i ? index : LIBXSMM_HASH_MOD(index + 1, LIBXSMM_REGSIZE)), \
              i = i0, next = LIBXSMM_HASH_MOD(i0 + 1, LIBXSMM_REGSIZE); \
              /* skip entries which correspond to no code, or continue on difference */ \
              (0 == (CODE = (internal_registry + i))->pmm || \
                0 != (diff = libxsmm_gemm_diff(DESCRIPTOR, &internal_registry_keys[i].descriptor))) \
                /* entire registry was searched and no code version was found */ \
                && next != i0; \
              i = next, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE)); \
            if (0 == diff) { /* found exact code version; continue with atomic load */ \
              flux_entry.pmm = (CODE)->pmm; \
              /* clear the collision and the non-JIT flag */ \
              flux_entry.imm &= ~(LIBXSMM_HASH_COLLISION | LIBXSMM_CODE_STATIC); \
            } \
            else { /* no code found */ \
              flux_entry.pmm = 0; \
            } \
            break; \
          } \
          else { /* clear the collision and the non-JIT flag */ \
            flux_entry.imm &= ~(LIBXSMM_HASH_COLLISION | LIBXSMM_CODE_STATIC); \
          } \
        } \
        else { /* new collision discovered (but no code version yet) */ \
          flux_entry.pmm = 0; \
        } \
      } \
      else { \
        diff = 0; \
      } \
      INTERNAL_FIND_CODE_JIT(DESCRIPTOR, CODE, flux_entry) \
      { \
        diff = 0; \
      } \
    } \
    while (0 != diff); \
    assert(0 == diff || 0 == flux_entry.pmm); \
    INTERNAL_FIND_CODE_CACHE_FINALIZE(cache_id, cache_keys, cache, cache_hit, flux_entry, DESCRIPTOR); \
  } \
} \
return flux_entry.xmm

#define INTERNAL_DISPATCH_MAIN(TYPE, DESCRIPTOR_DECL, DESC, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) { \
  INTERNAL_FIND_CODE_DECLARE(code); \
  const int internal_dispatch_main_flags_ = (0 == (PFLAGS) ? LIBXSMM_FLAGS : *(PFLAGS)) | LIBXSMM_GEMM_TYPEFLAG(TYPE); \
  const int internal_dispatch_main_lda_ = (0 == LIBXSMM_LD(PLDA, PLDB) ? LIBXSMM_LD(M, N) : *LIBXSMM_LD(PLDA, PLDB)); \
  const int internal_dispatch_main_ldb_ = (0 == LIBXSMM_LD(PLDB, PLDA) ? (K) : *LIBXSMM_LD(PLDB, PLDA)); \
  const int internal_dispatch_main_ldc_ = (0 == (PLDC) ? LIBXSMM_LD(M, N) : *(PLDC)); \
  const TYPE internal_dispatch_main_alpha_ = (0 == (PALPHA) ? ((TYPE)LIBXSMM_ALPHA) : *(PALPHA)); \
  const TYPE internal_dispatch_main_beta_ = (0 == (PBETA) ? ((TYPE)LIBXSMM_BETA) : *(PBETA)); \
  if (LIBXSMM_GEMM_NO_BYPASS(internal_dispatch_main_flags_, internal_dispatch_main_alpha_, internal_dispatch_main_beta_) && LIBXSMM_GEMM_NO_BYPASS_DIMS(M, N, K) && \
    LIBXSMM_GEMM_NO_BYPASS_DIMS(internal_dispatch_main_lda_, internal_dispatch_main_ldb_, internal_dispatch_main_ldc_)) \
  { \
    const int internal_dispatch_main_prefetch_ = (0 == (PREFETCH) \
      ? LIBXSMM_PREFETCH_NONE/*NULL-pointer is give; do not adopt LIBXSMM_PREFETCH (prefetches must be specified explicitly)*/ \
      : *(PREFETCH)); /*adopt the explicitly specified prefetch strategy*/ \
    DESCRIPTOR_DECL; LIBXSMM_GEMM_DESCRIPTOR(*(DESC), 0 != (VECTOR_WIDTH) ? (VECTOR_WIDTH): LIBXSMM_ALIGNMENT, \
      internal_dispatch_main_flags_, LIBXSMM_LD(M, N), LIBXSMM_LD(N, M), K, internal_dispatch_main_lda_, internal_dispatch_main_ldb_, internal_dispatch_main_ldc_, \
      (signed char)(internal_dispatch_main_alpha_), (signed char)(internal_dispatch_main_beta_), \
      (0 > internal_dispatch_main_prefetch_ ? libxsmm_gemm_auto_prefetch/*auto-dispatched*/ : internal_dispatch_main_prefetch_)); \
    { \
      INTERNAL_FIND_CODE(DESC, code).LIBXSMM_TPREFIX(TYPE, mm); \
    } \
  } \
  else { /* bypass (not supported) */ \
    /* libxsmm_gemm_print is not suitable here since A, B, and C are unknown at this point */ \
    libxsmm_update_mmstatistic(internal_dispatch_main_flags_, LIBXSMM_LD(M, N), LIBXSMM_LD(N, M), K, 1, 0); \
    return 0; \
  } \
}

#if defined(LIBXSMM_GEMM_DIFF_MASK_A) /* no padding i.e., LIBXSMM_GEMM_DESCRIPTOR_SIZE */
# define INTERNAL_DISPATCH(TYPE, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) \
    INTERNAL_DISPATCH_MAIN(TYPE, libxsmm_gemm_descriptor descriptor = { 0 }, &descriptor, \
    PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH)
#else /* padding: LIBXSMM_GEMM_DESCRIPTOR_SIZE -> LIBXSMM_ALIGNMENT */
# define INTERNAL_DISPATCH(TYPE, PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH) { \
    INTERNAL_DISPATCH_MAIN(TYPE, union { libxsmm_gemm_descriptor desc; char simd[LIBXSMM_ALIGNMENT]; } simd_descriptor; \
      for (i = LIBXSMM_GEMM_DESCRIPTOR_SIZE; i < sizeof(simd_descriptor.simd); ++i) simd_descriptor.simd[i] = 0, &simd_descriptor.desc, \
    PFLAGS, M, N, K, PLDA, PLDB, PLDC, PALPHA, PBETA, PREFETCH)
#endif

#if !defined(LIBXSMM_OPENMP) && !defined(LIBXSMM_NO_SYNC)
# define INTERNAL_REGLOCK_COUNT 16
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_LOCK_TYPE internal_reglock[INTERNAL_REGLOCK_COUNT];
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int internal_reglock_check;
#endif

LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE internal_regkey_type* internal_registry_keys /*= 0*/;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_code_pointer* internal_registry /*= 0*/;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE internal_statistic_type internal_statistic[2/*DP/SP*/][4/*sml/med/big/xxx*/];
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int internal_statistic_sml /*= 13*/;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int internal_statistic_med /*= 23*/;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int internal_statistic_mnk /*= LIBXSMM_MAX_M*/;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int internal_teardown /*= 0*/;


LIBXSMM_API_DEFINITION unsigned int libxsmm_update_mmstatistic(int flags, int m, int n, int k, unsigned int ntry, unsigned int ncol)
{
  const unsigned long long kernel_size = LIBXSMM_MNK_SIZE(m, n, k);
  const int precision = (0 == (LIBXSMM_GEMM_FLAG_F32PREC & flags) ? 0 : 1);
  int bucket = 3/*huge*/;

  if (LIBXSMM_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= kernel_size) {
    bucket = 0;
  }
  else if (LIBXSMM_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= kernel_size) {
    bucket = 1;
  }
  else if (LIBXSMM_MNK_SIZE(internal_statistic_mnk, internal_statistic_mnk, internal_statistic_mnk) >= kernel_size) {
    bucket = 2;
  }

  LIBXSMM_ATOMIC_ADD_FETCH(&internal_statistic[precision][bucket].ncol, ncol, LIBXSMM_ATOMIC_RELAXED);
  return LIBXSMM_ATOMIC_ADD_FETCH(&internal_statistic[precision][bucket].ntry, ntry, LIBXSMM_ATOMIC_RELAXED);
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE unsigned int internal_update_mmstatistic(const libxsmm_gemm_descriptor* desc,
  unsigned int ntry, unsigned int ncol)
{
  assert(0 != desc);
  return libxsmm_update_mmstatistic(desc->flags, desc->m, desc->n, desc->k, ntry, ncol);
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
    case LIBXSMM_X86_AVX512: {
      target_arch = "avx3";
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


LIBXSMM_INLINE LIBXSMM_RETARGETABLE unsigned int internal_print_number(unsigned int n, char default_unit, char* unit)
{
  unsigned int number = n;
  assert(0 != unit);
  *unit = default_unit;
  if ((1000000) <= n) {
    number = (n + 500000) / 1000000;
    *unit = 'm';
  }
  else if (9999 < n) {
    number = (n + 500) / 1000;
    *unit = 'k';
  }
  return number;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE unsigned int internal_print_statistic(FILE* ostream,
  const char* target_arch, int precision, unsigned int linebreaks, unsigned int indent)
{
  const internal_statistic_type statistic_sml = internal_statistic[precision][0/*SML*/];
  const internal_statistic_type statistic_med = internal_statistic[precision][1/*MED*/];
  const internal_statistic_type statistic_big = internal_statistic[precision][2/*BIG*/];
  const internal_statistic_type statistic_xxx = internal_statistic[precision][3/*XXX*/];
  int printed = 0;
  assert(0 != ostream && 0 != target_arch && (0 <= precision && precision < 2));

  if (/* omit to print anything if it is superfluous */
    0 != statistic_sml.ntry || 0 != statistic_sml.njit || 0 != statistic_sml.nsta || 0 != statistic_sml.ncol ||
    0 != statistic_med.ntry || 0 != statistic_med.njit || 0 != statistic_med.nsta || 0 != statistic_med.ncol ||
    0 != statistic_big.ntry || 0 != statistic_big.njit || 0 != statistic_big.nsta || 0 != statistic_big.ncol ||
    0 != statistic_xxx.ntry || 0 != statistic_xxx.njit || 0 != statistic_xxx.nsta || 0 != statistic_xxx.ncol)
  {
    char title[256], range[256], unit[4];
    unsigned int counter[4];
    {
      unsigned int n;
      assert(strlen(target_arch) < sizeof(title));
      for (n = 0; 0 != target_arch[n] /*avoid code-gen. issue with some clang versions: && n < sizeof(title)*/; ++n) {
        const char c = target_arch[n];
        title[n] = (char)(('a' <= c && c <= 'z') ? (c - 32) : c); /* toupper */
      }
      LIBXSMM_SNPRINTF(title + n, sizeof(title) - n, "/%s", 0 == precision ? "DP" : "SP");
      for (n = 0; n < linebreaks; ++n) fprintf(ostream, "\n");
    }
    fprintf(ostream, "%*s%-8s %6s %6s %6s %6s\n", (int)indent, "", title, "TRY" ,"JIT", "STA", "COL");
    LIBXSMM_SNPRINTF(range, sizeof(range), "%u..%u", 0u, internal_statistic_sml);
    counter[0] = internal_print_number(statistic_sml.ntry, ' ', unit + 0);
    counter[1] = internal_print_number(statistic_sml.njit, ' ', unit + 1);
    counter[2] = internal_print_number(statistic_sml.nsta, ' ', unit + 2);
    counter[3] = internal_print_number(statistic_sml.ncol, ' ', unit + 3);
    fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
      counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    LIBXSMM_SNPRINTF(range, sizeof(range), "%u..%u", internal_statistic_sml + 1u, internal_statistic_med);
    counter[0] = internal_print_number(statistic_med.ntry, ' ', unit + 0);
    counter[1] = internal_print_number(statistic_med.njit, ' ', unit + 1);
    counter[2] = internal_print_number(statistic_med.nsta, ' ', unit + 2);
    counter[3] = internal_print_number(statistic_med.ncol, ' ', unit + 3);
    fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
      counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    LIBXSMM_SNPRINTF(range, sizeof(range), "%u..%u", internal_statistic_med + 1u, internal_statistic_mnk);
    counter[0] = internal_print_number(statistic_big.ntry, ' ', unit + 0);
    counter[1] = internal_print_number(statistic_big.njit, ' ', unit + 1);
    counter[2] = internal_print_number(statistic_big.nsta, ' ', unit + 2);
    counter[3] = internal_print_number(statistic_big.ncol, ' ', unit + 3);
    fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
      counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    if (0 != statistic_xxx.ntry || 0 != statistic_xxx.njit || 0 != statistic_xxx.nsta || 0 != statistic_xxx.ncol) {
      LIBXSMM_SNPRINTF(range, sizeof(range), "> %u", internal_statistic_mnk);
      counter[0] = internal_print_number(statistic_xxx.ntry, ' ', unit + 0);
      counter[1] = internal_print_number(statistic_xxx.njit, ' ', unit + 1);
      counter[2] = internal_print_number(statistic_xxx.nsta, ' ', unit + 2);
      counter[3] = internal_print_number(statistic_xxx.ncol, ' ', unit + 3);
      fprintf(ostream, "%*s%8s %6u%c %5u%c %5u%c %5u%c\n", (int)indent, "", range,
        counter[0], unit[0], counter[1], unit[1], counter[2], unit[2], counter[3], unit[3]);
    }
    printed = 1;
  }

  return printed;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_register_static_code(const libxsmm_gemm_descriptor* desc,
  unsigned int index, unsigned int hash, libxsmm_xmmfunction src, libxsmm_code_pointer* registry)
{
  internal_regkey_type* dst_key = internal_registry_keys + index;
  libxsmm_code_pointer* dst_entry = registry + index;
#if !defined(NDEBUG)
  libxsmm_code_pointer code; code.xmm = src;
  assert(0 != desc && 0 != code.pmm && 0 != dst_key && 0 != registry);
  assert(0 == ((LIBXSMM_HASH_COLLISION | LIBXSMM_CODE_STATIC) & code.imm));
#endif

  if (0 != dst_entry->pmm) { /* collision? */
    /* start at a re-hashed index position */
    const unsigned int start = LIBXSMM_HASH_MOD(LIBXSMM_HASH_VALUE(hash), LIBXSMM_REGSIZE);
    unsigned int i0, i, next;

    /* mark current entry as a collision (this might be already the case) */
    dst_entry->imm |= LIBXSMM_HASH_COLLISION;

    /* start linearly searching for an available slot */
    for (i = (start != index) ? start : LIBXSMM_HASH_MOD(start + 1, LIBXSMM_REGSIZE), i0 = i, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE);
      0 != (dst_entry = registry + i)->pmm && next != i0; i = next, next = LIBXSMM_HASH_MOD(i + 1, LIBXSMM_REGSIZE));

    /* corresponding key position */
    dst_key = internal_registry_keys + i;

    internal_update_mmstatistic(desc, 0, 1);
  }

  if (0 == dst_entry->pmm) { /* registry not (yet) exhausted */
    dst_key->descriptor = *desc;
    dst_entry->xmm = src;
    /* mark current entry as a static (non-JIT) */
    dst_entry->imm |= LIBXSMM_CODE_STATIC;
  }

  internal_update_mmstatistic(desc, 1, 0);
}


LIBXSMM_API_DEFINITION int libxsmm_gemm_prefetch2uid(int prefetch)
{
  switch (prefetch) {
    case LIBXSMM_PREFETCH_SIGONLY:            return 2;
    case LIBXSMM_PREFETCH_BL2_VIA_C:          return 3;
    case LIBXSMM_PREFETCH_AL2_AHEAD:          return 4;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD: return 5;
    case LIBXSMM_PREFETCH_AL2:                return 6;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C:       return 7;
    case LIBXSMM_PREFETCH_AL2_JPST:           return 8;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST:  return 9;
    case LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C:    return 10;
    default: {
      assert(LIBXSMM_PREFETCH_NONE == prefetch);
      return 0;
    }
  }
}


LIBXSMM_API_DEFINITION int libxsmm_gemm_uid2prefetch(int uid)
{
  switch (uid) {
    case  2: return LIBXSMM_PREFETCH_SIGONLY;             /* pfsigonly */
    case  3: return LIBXSMM_PREFETCH_BL2_VIA_C;           /* BL2viaC */
    case  4: return LIBXSMM_PREFETCH_AL2_AHEAD;           /* curAL2 */
    case  5: return LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD;  /* curAL2_BL2viaC */
    case  6: return LIBXSMM_PREFETCH_AL2;                 /* AL2 */
    case  7: return LIBXSMM_PREFETCH_AL2BL2_VIA_C;        /* AL2_BL2viaC */
    case  8: return LIBXSMM_PREFETCH_AL2_JPST;            /* AL2jpst */
    case  9: return LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST;   /* AL2jpst_BL2viaC */
    case 10: return LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C;     /* AL2_BL2viaC_CL2 */
    default: return LIBXSMM_PREFETCH_NONE;
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE libxsmm_code_pointer* internal_init(void)
{
  /*const*/libxsmm_code_pointer* result;
  int i;
#if !defined(LIBXSMM_OPENMP)
# if !defined(LIBXSMM_NO_SYNC) /* setup the locks in a thread-safe fashion */
  assert(sizeof(internal_reglock) == (INTERNAL_REGLOCK_COUNT * sizeof(*internal_reglock)));
  if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&internal_reglock_check, 1, LIBXSMM_ATOMIC_SEQ_CST)) {
    for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXSMM_LOCK_INIT(internal_reglock + i);
    LIBXSMM_ATOMIC_STORE(&internal_reglock_check, 1, LIBXSMM_ATOMIC_SEQ_CST);
  }
  while (1 < internal_reglock_check) { /* wait until locks are initialized */
    if (1 == LIBXSMM_ATOMIC_LOAD(&internal_reglock_check, LIBXSMM_ATOMIC_SEQ_CST)) break;
  }
  for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXSMM_LOCK_ACQUIRE(internal_reglock + i);
# endif
#else
# pragma omp critical(internal_reglock)
#endif
  {
    result = LIBXSMM_ATOMIC_LOAD(&internal_registry, LIBXSMM_ATOMIC_SEQ_CST);
    if (0 == result) {
      int init_code = EXIT_FAILURE;
      libxsmm_set_target_arch(getenv("LIBXSMM_TARGET")); /* set libxsmm_target_archid */
      { /* select prefetch strategy for JIT */
        const char *const env = getenv("LIBXSMM_GEMM_PREFETCH");
        libxsmm_gemm_auto_prefetch = (LIBXSMM_X86_AVX512_MIC != libxsmm_target_archid ? INTERNAL_PREFETCH : LIBXSMM_PREFETCH_AL2BL2_VIA_C);
        if (0 != env && 0 != *env) { /* user input beyond auto-prefetch is always considered */
          const int uid = atoi(env);
          if (0 <= uid) {
            libxsmm_gemm_auto_prefetch = libxsmm_gemm_uid2prefetch(uid);
          }
        }
      }
      libxsmm_mt = 2;
      {
        /* behaviour of parallelized routines which are located in libxsmmext library
         * 0: sequential below-threshold routine (no OpenMP); may fall-back to BLAS,
         * 1: (OpenMP-)parallelized but without internal parallel region,
         * 2: (OpenMP-)parallelized with internal parallel region"
         */
        const char *const env = getenv("LIBXSMM_MT");
        if (0 != env && 0 != *env) {
          libxsmm_mt = atoi(env);
        }
      }
      {
        const char *const env = getenv("LIBXSMM_TASKS");
        if (0 != env && 0 != *env) {
          libxsmm_tasks = atoi(env);
        }
      }
      libxsmm_nt = 2;
#if !defined(__MIC__) && (LIBXSMM_X86_AVX512_MIC != LIBXSMM_STATIC_TARGET_ARCH)
      if (LIBXSMM_X86_AVX512_MIC == libxsmm_target_archid)
#endif
      {
        libxsmm_nt = 4;
      }
      {
        const char *const env = getenv("LIBXSMM_VERBOSE");
        internal_statistic_mnk = (unsigned int)(pow((double)(LIBXSMM_MAX_MNK), 0.3333333333333333) + 0.5);
        internal_statistic_sml = 13; internal_statistic_med = 23;
        if (0 != env && 0 != *env) {
          libxsmm_verbosity = atoi(env);
        }
#if !defined(NDEBUG)
        else {
          libxsmm_verbosity = 1; /* quiet -> verbose */
        }
#endif
      }
#if !defined(__TRACE)
      LIBXSMM_UNUSED(init_code);
#else
      {
        int filter_threadid = 0, filter_mindepth = 1, filter_maxnsyms = 0;
        const char *const env = getenv("LIBXSMM_TRACE");
        if (0 != env && 0 != *env) {
          char buffer[32];
          if (1 == sscanf(env, "%32[^,],", buffer)) {
            sscanf(buffer, "%i", &filter_threadid);
          }
          if (1 == sscanf(env, "%*[^,],%32[^,],", buffer)) {
            sscanf(buffer, "%i", &filter_mindepth);
          }
          if (1 == sscanf(env, "%*[^,],%*[^,],%32s", buffer)) {
            sscanf(buffer, "%i", &filter_maxnsyms);
          }
          else {
            filter_maxnsyms = -1; /* all */
          }
        }
        init_code = libxsmm_trace_init(filter_threadid - 1, filter_mindepth, filter_maxnsyms);
      }
      if (EXIT_SUCCESS == init_code)
#endif
      {
        libxsmm_gemm_init(libxsmm_target_archid, libxsmm_gemm_auto_prefetch);
        libxsmm_gemm_diff_init(libxsmm_target_archid);
        libxsmm_trans_init(libxsmm_target_archid);
        libxsmm_hash_init(libxsmm_target_archid);
#if defined(LIBXSMM_PERF)
        libxsmm_perf_init();
#endif
        assert(0 == internal_registry_keys && 0 == internal_registry); /* should never happen */
        result = (libxsmm_code_pointer*)libxsmm_malloc(LIBXSMM_REGSIZE * sizeof(libxsmm_code_pointer));
        internal_registry_keys = (internal_regkey_type*)libxsmm_malloc(LIBXSMM_REGSIZE * sizeof(internal_regkey_type));
        if (0 != result && 0 != internal_registry_keys) {
          for (i = 0; i < LIBXSMM_REGSIZE; ++i) result[i].pmm = 0;
          /* omit registering code if JIT is enabled and if an ISA extension is found
           * which is beyond the static code path used to compile the library
           */
#if defined(LIBXSMM_BUILD)
# if (0 != LIBXSMM_JIT) && !defined(__MIC__)
          /* check if target arch. permits execution (arch. may be overridden) */
          if (LIBXSMM_STATIC_TARGET_ARCH <= libxsmm_target_archid &&
             (LIBXSMM_X86_AVX > libxsmm_target_archid /* jit is not available */
              /* condition allows to avoid JIT (if static code is good enough) */
           || LIBXSMM_STATIC_TARGET_ARCH == libxsmm_target_archid))
# endif
          { /* opening a scope for eventually declaring variables */
            /* setup the dispatch table for the statically generated code */
#           include <libxsmm_dispatch.h>
          }
#endif
          atexit(libxsmm_finalize);
          {
            void *const pv_registry = &internal_registry;
            LIBXSMM_ATOMIC_STORE((void**)pv_registry, (void*)result, LIBXSMM_ATOMIC_SEQ_CST);
          }
        }
        else {
#if !defined(NDEBUG) && defined(__TRACE) /* library code is expected to be mute */
          fprintf(stderr, "LIBXSMM: failed to allocate code registry!\n");
#endif
          libxsmm_free(internal_registry_keys);
          libxsmm_free(result);
        }
      }
#if !defined(NDEBUG) && defined(__TRACE) /* library code is expected to be mute */
      else {
        fprintf(stderr, "LIBXSMM: failed to initialize TRACE (error #%i)!\n", init_code);
      }
#endif
    }
  }
#if !defined(LIBXSMM_OPENMP) && !defined(LIBXSMM_NO_SYNC) /* release locks */
  for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXSMM_LOCK_RELEASE(internal_reglock + i);
#endif
  assert(result);
  return result;
}


LIBXSMM_API_DEFINITION LIBXSMM_CTOR_ATTRIBUTE void libxsmm_init(void)
{
  const void *const registry = LIBXSMM_ATOMIC_LOAD(&internal_registry, LIBXSMM_ATOMIC_RELAXED);
  if (0 == registry) {
    internal_init();
  }
}


LIBXSMM_API
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void libxsmm_finalize(void);

LIBXSMM_API_DEFINITION LIBXSMM_DTOR_ATTRIBUTE void libxsmm_finalize(void)
{
  libxsmm_code_pointer* registry = LIBXSMM_ATOMIC_LOAD(&internal_registry, LIBXSMM_ATOMIC_SEQ_CST);

  if (0 != registry) {
    int i;
#if !defined(LIBXSMM_OPENMP)
# if !defined(LIBXSMM_NO_SYNC)
    /* acquire locks and thereby shortcut lazy initialization later on */
    for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXSMM_LOCK_ACQUIRE(internal_reglock + i);
# endif
#else
#   pragma omp critical(internal_reglock)
#endif
    {
      registry = internal_registry;

      if (0 != registry) {
        internal_regkey_type *const registry_keys = internal_registry_keys;
        const char *const target_arch = internal_get_target_arch(libxsmm_target_archid);
        unsigned int heapmem = (LIBXSMM_REGSIZE) * (sizeof(libxsmm_code_pointer) + sizeof(internal_regkey_type));

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
        libxsmm_trans_finalize();
        libxsmm_hash_finalize();
#if defined(LIBXSMM_PERF)
        libxsmm_perf_finalize();
#endif
        /* make internal registry globally unavailable */
        LIBXSMM_ATOMIC_STORE_ZERO(&internal_registry, LIBXSMM_ATOMIC_SEQ_CST);
        internal_registry_keys = 0;

        for (i = 0; i < LIBXSMM_REGSIZE; ++i) {
          libxsmm_code_pointer code = registry[i];
          if (0 != code.pmm) {
            const libxsmm_gemm_descriptor *const desc = &registry_keys[i].descriptor;
            const unsigned long long kernel_size = LIBXSMM_MNK_SIZE(desc->m, desc->n, desc->k);
            const int precision = (0 == (LIBXSMM_GEMM_FLAG_F32PREC & desc->flags) ? 0 : 1);
            int bucket = 3/*huge*/;
            assert((LIBXSMM_HASH_COLLISION | LIBXSMM_CODE_STATIC) != code.imm);
            if (LIBXSMM_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= kernel_size) {
              bucket = 0;
            }
            else if (LIBXSMM_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= kernel_size) {
              bucket = 1;
            }
            else if (LIBXSMM_MNK_SIZE(internal_statistic_mnk, internal_statistic_mnk, internal_statistic_mnk) >= kernel_size) {
              bucket = 2;
            }
            if (0 == (LIBXSMM_CODE_STATIC & code.imm)) { /* check for allocated/generated JIT-code */
              void* buffer = 0;
              size_t size = 0;
              code.imm &= ~LIBXSMM_HASH_COLLISION; /* clear collision flag */
              if (EXIT_SUCCESS == libxsmm_malloc_info(code.pmm, &size, 0/*flags*/, &buffer)) {
                libxsmm_xfree(code.pmm);
                ++internal_statistic[precision][bucket].njit;
                heapmem += (unsigned int)(size + (((char*)code.pmm) - (char*)buffer));
              }
            }
            else {
              ++internal_statistic[precision][bucket].nsta;
            }
          }
        }
        if (0 != libxsmm_verbosity) { /* print statistic on termination */
          LIBXSMM_FLOCK(stderr);
          LIBXSMM_FLOCK(stdout);
          fflush(stdout); /* synchronize with standard output */
          {
            const unsigned int linebreak = (0 == internal_print_statistic(stderr, target_arch, 1/*SP*/, 1, 0)) ? 1 : 0;
            if (0 == internal_print_statistic(stderr, target_arch, 0/*DP*/, linebreak, 0) && 0 != linebreak) {
              fprintf(stderr, "LIBXSMM_TARGET=%s ", target_arch);
            }
            fprintf(stderr, "HEAP: %.f MB\n", 1.0 * heapmem / (1 << 20));
          }
          LIBXSMM_FUNLOCK(stdout);
          LIBXSMM_FUNLOCK(stderr);
        }
        libxsmm_free(registry_keys);
        libxsmm_free(registry);
      }
    }
#if !defined(LIBXSMM_OPENMP) && !defined(LIBXSMM_NO_SYNC) /* release locks */
    for (i = 0; i < INTERNAL_REGLOCK_COUNT; ++i) LIBXSMM_LOCK_RELEASE(internal_reglock + i);
    LIBXSMM_ATOMIC_STORE_ZERO(&internal_reglock_check, LIBXSMM_ATOMIC_SEQ_CST);
#endif
  }
}


LIBXSMM_API_DEFINITION int libxsmm_get_target_archid(void)
{
  LIBXSMM_INIT
#if !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  return libxsmm_target_archid;
#else /* no JIT support */
  return LIBXSMM_MIN(libxsmm_target_archid, LIBXSMM_X86_SSE4_2);
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_set_target_archid(int id)
{
  int target_archid = LIBXSMM_TARGET_ARCH_UNKNOWN;
  switch (id) {
    case LIBXSMM_X86_AVX512_CORE:
    case LIBXSMM_X86_AVX512_MIC:
    case LIBXSMM_X86_AVX512:
    case LIBXSMM_X86_AVX2:
    case LIBXSMM_X86_AVX:
    case LIBXSMM_X86_SSE4_2:
    case LIBXSMM_X86_SSE4_1:
    case LIBXSMM_X86_SSE3:
    case LIBXSMM_TARGET_ARCH_GENERIC: {
      target_archid = id;
    } break;
    default: if (LIBXSMM_X86_GENERIC <= id) {
      target_archid = LIBXSMM_X86_GENERIC;
    }
    else {
      target_archid = libxsmm_cpuid();
    }
  }
  LIBXSMM_ATOMIC_STORE(&libxsmm_target_archid, target_archid, LIBXSMM_ATOMIC_RELAXED);
#if !defined(NDEBUG) /* library code is expected to be mute */
  {
    const int cpuid = libxsmm_cpuid();
    if (cpuid < libxsmm_target_archid) {
      const char *const target_arch = internal_get_target_arch(libxsmm_target_archid);
      fprintf(stderr, "LIBXSMM: \"%s\" code will fail to run on \"%s\"!\n",
        target_arch, internal_get_target_arch(cpuid));
    }
  }
#endif
}


LIBXSMM_API_DEFINITION const char* libxsmm_get_target_arch(void)
{
  LIBXSMM_INIT
  return internal_get_target_arch(libxsmm_target_archid);
}


/* function serves as a helper for implementing the Fortran interface */
LIBXSMM_API const char* get_target_arch(int* length);
LIBXSMM_API_DEFINITION const char* get_target_arch(int* length)
{
  const char *const arch = libxsmm_get_target_arch();
  /* valid here since function is not in the public interface */
  assert(0 != arch && 0 != length);
  *length = (int)strlen(arch);
  return arch;
}


LIBXSMM_API_DEFINITION void libxsmm_set_target_arch(const char* arch)
{
  int target_archid = LIBXSMM_TARGET_ARCH_UNKNOWN;
  if (0 != arch && 0 != *arch) {
    const int jit = atoi(arch);
    if (0 == strcmp("0", arch)) {
      target_archid = LIBXSMM_TARGET_ARCH_GENERIC;
    }
    else if (1 < jit) {
      target_archid = LIBXSMM_X86_GENERIC + jit;
    }
    else if (0 == strcmp("skx", arch) || 0 == strcmp("skl", arch)) {
      target_archid = LIBXSMM_X86_AVX512_CORE;
    }
    else if (0 == strcmp("knl", arch) || 0 == strcmp("mic", arch)) {
      target_archid = LIBXSMM_X86_AVX512_MIC;
    }
    else if (0 == strcmp("avx3", arch) || 0 == strcmp("avx512", arch)) {
      target_archid = LIBXSMM_X86_AVX512;
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
    target_archid = libxsmm_cpuid();
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    const int cpuid = libxsmm_cpuid();
    if (cpuid < target_archid) {
      const char *const target_arch = internal_get_target_arch(target_archid);
      fprintf(stderr, "LIBXSMM: \"%s\" code will fail to run on \"%s\"!\n",
        target_arch, internal_get_target_arch(cpuid));
    }
  }
#endif
  LIBXSMM_ATOMIC_STORE(&libxsmm_target_archid, target_archid, LIBXSMM_ATOMIC_RELAXED);
}


LIBXSMM_API_DEFINITION int libxsmm_get_verbosity(void)
{
  LIBXSMM_INIT
  return libxsmm_verbosity;
}


LIBXSMM_API_DEFINITION void libxsmm_set_verbosity(int level)
{
  LIBXSMM_INIT
  LIBXSMM_ATOMIC_STORE(&libxsmm_verbosity, level, LIBXSMM_ATOMIC_RELAXED);
}


LIBXSMM_API_DEFINITION libxsmm_gemm_prefetch_type libxsmm_get_gemm_auto_prefetch(void)
{
  return (libxsmm_gemm_prefetch_type)libxsmm_gemm_auto_prefetch;
}


LIBXSMM_API_DEFINITION void libxsmm_set_gemm_auto_prefetch(libxsmm_gemm_prefetch_type strategy)
{
  LIBXSMM_ATOMIC_STORE(&libxsmm_gemm_auto_prefetch, strategy, LIBXSMM_ATOMIC_RELAXED);
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE const char* internal_get_precision_string(libxsmm_dnn_datatype datatype)
{
  const char* result = "unk"; /* unknown */
  switch (datatype) {
    case LIBXSMM_DNN_DATATYPE_F32: result = "f32"; break;
    case LIBXSMM_DNN_DATATYPE_I32: result = "i32"; break;
    case LIBXSMM_DNN_DATATYPE_I16: result = "i16"; break;
    case LIBXSMM_DNN_DATATYPE_I8:  result = "i8";  break;
  }
  return result;
}


LIBXSMM_API_DEFINITION void libxsmm_build(const libxsmm_build_request* request, unsigned int regindex, libxsmm_code_pointer* code)
{
#if !defined(__MIC__) && (!defined(__CYGWIN__) || !defined(NDEBUG)/*code-coverage with Cygwin; fails@runtime!*/)
  const char *const target_arch = internal_get_target_arch(libxsmm_target_archid);
  libxsmm_generated_code generated_code;
  char jit_name[256] = { 0 };

  assert(0 != request && 0 != libxsmm_target_archid);
  assert(0 != code && 0 == code->pmm);
  /* some shared setup for code generation */
  memset(&generated_code, 0, sizeof(generated_code));
  generated_code.code_size = 0;
  generated_code.code_type = 2;
  generated_code.last_error = 0;

  switch (request->kind) { /* generate kernel */
    case LIBXSMM_BUILD_KIND_GEMM: { /* small MxM kernel */
      assert(0 != request->descriptor.gemm);
      if (0 < request->descriptor.gemm->m   && 0 < request->descriptor.gemm->n   && 0 < request->descriptor.gemm->k &&
          0 < request->descriptor.gemm->lda && 0 < request->descriptor.gemm->lda && 0 < request->descriptor.gemm->ldc)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_gemm_kernel, &generated_code, request->descriptor.gemm, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid(request->descriptor.gemm->prefetch);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.mxm", target_arch/*code path name*/,
            0 == (LIBXSMM_GEMM_FLAG_F32PREC & request->descriptor.gemm->flags) ? "f64" : "f32",
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.gemm->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.gemm->m,   (unsigned int)request->descriptor.gemm->n,   (unsigned int)request->descriptor.gemm->k,
            (unsigned int)request->descriptor.gemm->lda, (unsigned int)request->descriptor.gemm->ldb, (unsigned int)request->descriptor.gemm->ldc,
            request->descriptor.gemm->alpha, request->descriptor.gemm->beta, uid);
        }
      }
      else { /* this case is not an actual error */
        return;
      }
    } break;
    case LIBXSMM_BUILD_KIND_SSOA: { /* sparse SOA kernel */
      assert(0 != request->descriptor.ssoa && 0 != request->descriptor.ssoa->gemm);
      assert(0 != request->descriptor.ssoa->row_ptr && 0 != request->descriptor.ssoa->column_idx && 0 != request->descriptor.ssoa->values);
      if (0 == (LIBXSMM_GEMM_FLAG_F32PREC & (request->descriptor.ssoa->gemm->flags))/*only double-precision*/) {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_spgemm_csr_soa_kernel, &generated_code, request->descriptor.ssoa->gemm, target_arch,
          request->descriptor.ssoa->row_ptr, request->descriptor.ssoa->column_idx,
          (const double*)request->descriptor.ssoa->values);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid(request->descriptor.ssoa->gemm->prefetch);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.ssoa", target_arch/*code path name*/,
            0 == (LIBXSMM_GEMM_FLAG_F32PREC & request->descriptor.ssoa->gemm->flags) ? "f64" : "f32",
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.ssoa->gemm->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.ssoa->gemm->flags) ? 'n' : 't',
            (unsigned int)request->descriptor.ssoa->gemm->m,   (unsigned int)request->descriptor.ssoa->gemm->n,   (unsigned int)request->descriptor.ssoa->gemm->k,
            (unsigned int)request->descriptor.ssoa->gemm->lda, (unsigned int)request->descriptor.ssoa->gemm->ldb, (unsigned int)request->descriptor.ssoa->gemm->ldc,
            request->descriptor.ssoa->gemm->alpha, request->descriptor.ssoa->gemm->beta, uid);
        }
      }
      else { /* this case is not an actual error */
        return;
      }
    } break;
    case LIBXSMM_BUILD_KIND_CFWD: { /* forward convolution */
      assert(0 != request->descriptor.cfwd);
      if (0 < request->descriptor.cfwd->kw && 0 < request->descriptor.cfwd->kh &&
          0 != request->descriptor.cfwd->stride_w && 0 != request->descriptor.cfwd->stride_h)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_convolution_forward_kernel, &generated_code, request->descriptor.cfwd, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(request->descriptor.cfwd->datatype_in);
          const char *const precision_out = internal_get_precision_string(request->descriptor.cfwd->datatype_out);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_fwd_%s_%s_%ux%u_%ux%uu_s%ii%io_vl%ui%uo_ri%ux%u_ro%ux%u_r%ux%u_p%i_f%i.conv",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cfwd->kw/*kernel width*/, (unsigned int)request->descriptor.cfwd->kh/*kernel height*/,
            (unsigned int)request->descriptor.cfwd->unroll_kw/*width*/, (unsigned int)request->descriptor.cfwd->unroll_kh/*height*/,
            (int)request->descriptor.cfwd->stride_w/*input offset*/, (int)request->descriptor.cfwd->stride_h/*output offsets*/,
            (unsigned int)request->descriptor.cfwd->ifm_block/*VLEN*/, (unsigned int)request->descriptor.cfwd->ofm_block/*VLEN*/,
            (unsigned int)request->descriptor.cfwd->ifw_padded, (unsigned int)request->descriptor.cfwd->ifh_padded,
            (unsigned int)request->descriptor.cfwd->ofw_padded/*1D and 2D register block*/,
            (unsigned int)request->descriptor.cfwd->ofh_padded/*2D register block*/,
            (unsigned int)request->descriptor.cfwd->ofw_rb/*register block ofw*/,
            (unsigned int)request->descriptor.cfwd->ofh_rb/*register block ofh*/,
            (int)request->descriptor.cfwd->prefetch/*binary OR'd prefetch flags*/,
            (int)request->descriptor.cfwd->format/*binary OR'd format flags*/);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_CBWD: { /* backward convolution */
      assert(0 != request->descriptor.cbwd);
      if (0 < request->descriptor.cbwd->kw && 0 < request->descriptor.cbwd->kh &&
          0 != request->descriptor.cbwd->stride_w && 0 != request->descriptor.cbwd->stride_h)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_convolution_backward_kernel, &generated_code, request->descriptor.cbwd, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(request->descriptor.cbwd->datatype_in);
          const char *const precision_out = internal_get_precision_string(request->descriptor.cbwd->datatype_out);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_bwd_%s_%s_%ux%u_%ux%uu_s%ii%io_vl%ui%uo_ri%ux%u_ro%ux%u_r%ux%u_of%uu%u_v%u_pa%u_p%i_f%i.conv",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cbwd->kw/*kernel width*/, (unsigned int)request->descriptor.cbwd->kh/*kernel height*/,
            (unsigned int)request->descriptor.cbwd->unroll_kw/*width*/, (unsigned int)request->descriptor.cbwd->unroll_kh/*height*/,
            (int)request->descriptor.cbwd->stride_w/*input offset*/, (int)request->descriptor.cbwd->stride_h/*output offsets*/,
            (unsigned int)request->descriptor.cbwd->ifm_block/*VLEN*/, (unsigned int)request->descriptor.cbwd->ofm_block/*VLEN*/,
            (unsigned int)request->descriptor.cbwd->ifw_padded, (unsigned int)request->descriptor.cbwd->ifh_padded,
            (unsigned int)request->descriptor.cbwd->ofw_padded/*1D and 2D register block*/,
            (unsigned int)request->descriptor.cbwd->ofh_padded/*2D register block*/,
            (unsigned int)request->descriptor.cbwd->ofw_rb/*register block ofw*/,
            (unsigned int)request->descriptor.cbwd->ofh_rb/*register block ofh*/,
            (unsigned int)request->descriptor.cbwd->ofw/*ofw*/, (unsigned int)request->descriptor.cbwd->ofw_unroll/*ofw_unroll*/,
            (unsigned int)request->descriptor.cbwd->peeled/*peeled version*/,
            (unsigned int)request->descriptor.cbwd->prefetch_output_ahead/*prefetch kj outputs for jumping from non-peel to peel version*/,
            (int)request->descriptor.cbwd->prefetch/*binary OR'd prefetch flags*/,
            (int)request->descriptor.cbwd->format/*binary OR'd format flags*/);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_CUPD: { /* convolution update weights */
      assert(0 != request->descriptor.cupd);
      if (0 < request->descriptor.cupd->kw &&
          0 != request->descriptor.cupd->stride_w && 0 != request->descriptor.cupd->stride_h)
      {
        generated_code.generated_code = malloc(131072); /* large enough temporary buffer for generated code */
        generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_convolution_weight_update_kernel, &generated_code, request->descriptor.cupd, target_arch);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const char *const precision_in = internal_get_precision_string(request->descriptor.cupd->datatype_in);
          const char *const precision_out = internal_get_precision_string(request->descriptor.cupd->datatype_out);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_upd_%s_%s_%ux%u_%uu_s%ii%io_vl%ui%uo_ri%ux%u_ro%ux%u_r%ux%u_of%uu%ux%uu%u_if%uu_t%u_p%i_f%i.conv",
            target_arch/*code path name*/, precision_in, precision_out,
            (unsigned int)request->descriptor.cupd->kw/*kernel width*/, (unsigned int)request->descriptor.cupd->kh/*kernel height*/,
            (unsigned int)request->descriptor.cupd->unroll_kw/*width*/,
            (int)request->descriptor.cupd->stride_w/*input offset*/, (int)request->descriptor.cupd->stride_h/*output offsets*/,
            (unsigned int)request->descriptor.cupd->ifm_block/*VLEN*/, (unsigned int)request->descriptor.cupd->ofm_block/*VLEN*/,
            (unsigned int)request->descriptor.cupd->ifw_padded, (unsigned int)request->descriptor.cupd->ifh_padded,
            (unsigned int)request->descriptor.cupd->ofw_padded/*1D and 2D register block*/,
            (unsigned int)request->descriptor.cupd->ofh_padded/*2D register block*/,
            (unsigned int)request->descriptor.cupd->ofw_rb/*register block ofw*/,
            (unsigned int)request->descriptor.cupd->ofh_rb/*register block ofh*/,
            (unsigned int)request->descriptor.cupd->ofw/*ofw*/, (unsigned int)request->descriptor.cupd->ofw_unroll/*ofw_unroll*/,
            (unsigned int)request->descriptor.cupd->ofh/*ofh*/, (unsigned int)request->descriptor.cupd->ofh_unroll/*ofh_unroll*/,
            (unsigned int)request->descriptor.cupd->ifm_unroll/*ifm unroll*/,
            (unsigned int)request->descriptor.cupd->transpose_ofw_ifm/*transpose_ofw_ifm*/,
            (int)request->descriptor.cupd->prefetch/*binary OR'd prefetch flags*/,
            (int)request->descriptor.cupd->format/*binary OR'd format flags*/);
        }
      }
    } break;
# if !defined(NDEBUG) /* library code is expected to be mute */
    default: { /* unknown kind */
      static int error_once = 0;
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXSMM: invalid build request discovered!\n");
      }
    }
# endif
  }

  /* handle an eventual error in the else-branch */
  if (0 == generated_code.last_error) {
    /* attempt to create executable buffer, and check for success */
    if (0 < generated_code.code_size && /* check for previous match (build kind) */
      EXIT_SUCCESS == libxsmm_xmalloc(&code->pmm, generated_code.code_size, 0/*auto*/,
        /* flag must be a superset of what's populated by libxsmm_malloc_attrib */
        LIBXSMM_MALLOC_FLAG_RWX, &regindex, sizeof(regindex)))
    {
      assert(0 != code->pmm && 0 == ((LIBXSMM_HASH_COLLISION | LIBXSMM_CODE_STATIC) & code->imm));
      /* copy temporary buffer into the prepared executable buffer */
      memcpy(code->pmm, generated_code.generated_code, generated_code.code_size);
      /* attribute/protect buffer and revoke unnecessary flags; continue on error */
      libxsmm_malloc_attrib(&code->pmm, LIBXSMM_MALLOC_FLAG_X, jit_name);
    }
  }
# if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      LIBXSMM_NO_OFFLOAD(int, fprintf, stderr, "%s (error #%u)\n",
        LIBXSMM_NO_OFFLOAD(const char*, libxsmm_strerror, generated_code.last_error),
        generated_code.last_error);
    }
  }
# endif
  free(generated_code.generated_code); /* free temporary/initial code buffer */
#else /* unsupported platform */
  LIBXSMM_UNUSED(request); LIBXSMM_UNUSED(regindex); LIBXSMM_UNUSED(code);
  /* libxsmm_get_target_arch also serves as a runtime check whether JIT is available or not */
  assert(LIBXSMM_X86_AVX > libxsmm_target_archid);
#endif
}


LIBXSMM_API_DEFINITION libxsmm_xmmfunction libxsmm_xmmdispatch(const libxsmm_gemm_descriptor* descriptor)
{
  const libxsmm_xmmfunction null_mmfunction = { 0 };
  /* there is no need to check LIBXSMM_GEMM_NO_BYPASS_DIMS (M, N, K, LDx) since we already got a descriptor */
  if (0 != descriptor && LIBXSMM_GEMM_NO_BYPASS(descriptor->flags, descriptor->alpha, descriptor->beta)) {
    libxsmm_gemm_descriptor backend_descriptor;
    LIBXSMM_INIT
    if (0 > (int)descriptor->prefetch) {
      backend_descriptor = *descriptor;
      backend_descriptor.prefetch = (unsigned char)libxsmm_gemm_auto_prefetch;
      descriptor = &backend_descriptor;
    }
    {
      INTERNAL_FIND_CODE_DECLARE(code);
      INTERNAL_FIND_CODE(descriptor, code);
    }
  }
  else { /* bypass (not supported) */
    internal_update_mmstatistic(descriptor, 1, 0);
    return null_mmfunction;
  }
}

#if !defined(LIBXSMM_BUILD) && defined(__APPLE__) && defined(__MACH__)
LIBXSMM_PRAGMA_OPTIMIZE_OFF
#endif

LIBXSMM_API_DEFINITION libxsmm_smmfunction libxsmm_smmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch)
{
  LIBXSMM_INIT
  INTERNAL_DISPATCH(float, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}


LIBXSMM_API_DEFINITION libxsmm_dmmfunction libxsmm_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch)
{
  LIBXSMM_INIT
  INTERNAL_DISPATCH(double, flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
}

#if !defined(LIBXSMM_BUILD) && defined(__APPLE__) && defined(__MACH__)
LIBXSMM_PRAGMA_OPTIMIZE_ON
#endif

LIBXSMM_API_DEFINITION libxsmm_xmmfunction libxsmm_create_dcsr_soa(const libxsmm_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values)
{
  libxsmm_code_pointer code = { 0 };
  libxsmm_csr_soa_descriptor ssoa;
  libxsmm_build_request request;
  LIBXSMM_INIT
  ssoa.gemm = descriptor;
  ssoa.row_ptr = row_ptr;
  ssoa.column_idx = column_idx;
  ssoa.values = values;
  request.descriptor.ssoa = &ssoa;
  request.kind = LIBXSMM_BUILD_KIND_SSOA;
  libxsmm_build(&request, LIBXSMM_REGSIZE/*not managed*/, &code);
  return code.xmm;
}


LIBXSMM_API_DEFINITION void libxsmm_release_kernel(const void* jit_code)
{
  void* extra = 0;
  LIBXSMM_INIT
  if (EXIT_SUCCESS == libxsmm_malloc_info((const volatile void*)jit_code, 0/*size*/, 0/*flags*/, &extra) && 0 != extra) {
    const unsigned int regindex = *((const unsigned int*)extra);
    if (LIBXSMM_REGSIZE <= regindex) {
      libxsmm_xfree((const volatile void*)jit_code);
    }
    /* TODO: implement to unregister GEMM kernels */
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: failed to release kernel!\n");
    }
  }
#endif
}

