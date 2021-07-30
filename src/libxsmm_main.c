/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_trace.h"
#include "libxsmm_xcopy.h"
#include "libxsmm_gemm.h"
#include "libxsmm_hash.h"
#include "libxsmm_diff.h"
#include "libxsmm_main.h"
#if defined(LIBXSMM_PERF)
# include "libxsmm_perf.h"
#endif
#include "generator_common.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(NDEBUG)
# include <errno.h>
#endif
#if defined(_WIN32)
# include <Windows.h>
#else
# include <sys/types.h>
# include <sys/mman.h>
# include <sys/stat.h>
# include <unistd.h>
# include <fcntl.h>
#endif
#if defined(__APPLE__)
# include <libkern/OSCacheControl.h>
# include <pthread.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_CODE_MAXSIZE)
# define LIBXSMM_CODE_MAXSIZE 131072
#endif
#if !defined(LIBXSMM_DIFF_SIZE)
# define LIBXSMM_DIFF_SIZE LIBXSMM_DESCRIPTOR_SIGSIZE
#endif
#if !defined(LIBXSMM_HASH_SIZE)
/* can be smaller than MAXSIZE/SIGSIZE at the expense of collisions */
# define LIBXSMM_HASH_SIZE 32
#endif
#if !defined(LIBXSMM_HASH_SEED)
# define LIBXSMM_HASH_SEED 25071975
#endif
#if !defined(LIBXSMM_MALLOC_HOOK_ALIGN) && 1
# define LIBXSMM_MALLOC_HOOK_ALIGN
#endif
#if !defined(LIBXSMM_ENABLE_DEREG) && 0
# define LIBXSMM_ENABLE_DEREG
#endif
#if !defined(LIBXSMM_REGUSER_HASH) && 1
# define LIBXSMM_REGUSER_HASH
#endif
#if !defined(LIBXSMM_REGLOCK_TRY) && 0
# define LIBXSMM_REGLOCK_TRY
#endif
#if !defined(LIBXSMM_UNIFY_LOCKS) && 1
# define LIBXSMM_UNIFY_LOCKS
#endif
#if !defined(LIBXSMM_REGKEY_PAD) && 0
# define LIBXSMM_REGKEY_PAD
#endif
#if !defined(LIBXSMM_CACHE_PAD) && 1
# define LIBXSMM_CACHE_PAD
#endif
#if !defined(LIBXSMM_AUTOPIN) && 0
# define LIBXSMM_AUTOPIN
#endif
#if !defined(INTERNAL_DELIMS)
# define INTERNAL_DELIMS ";,:"
#endif

#if !defined(_WIN32) && !defined(__CYGWIN__)
LIBXSMM_EXTERN int posix_memalign(void**, size_t, size_t) LIBXSMM_THROW;
#endif
#if defined(LIBXSMM_AUTOPIN) && !defined(_WIN32)
LIBXSMM_EXTERN int putenv(char*) LIBXSMM_THROW;
#endif

/* flag fused into the memory address of a code version in case of non-JIT */
#define LIBXSMM_CODE_STATIC (1ULL << (8 * sizeof(void*) - 1))
/* flag fused into the memory address of a code version in case of collision */
#if 1 /* beneficial when registry approaches capacity (collisions) */
# define LIBXSMM_HASH_COLLISION (1ULL << (8 * sizeof(void*) - 2))
#endif

/** Helper macro determining the default prefetch strategy which is used for statically generated kernels. */
#if (0 > LIBXSMM_PREFETCH) /* auto-prefetch (frontend) */ || (defined(_WIN32) || defined(__CYGWIN__))
# define INTERNAL_PREFETCH LIBXSMM_GEMM_PREFETCH_NONE
#else
# define INTERNAL_PREFETCH ((libxsmm_gemm_prefetch_type)LIBXSMM_PREFETCH)
#endif

#if (0 != LIBXSMM_SYNC)
# if !defined(INTERNAL_REGLOCK_MAXN)
#   if defined(_MSC_VER)
#     define INTERNAL_REGLOCK_MAXN 0
#   else
#     define INTERNAL_REGLOCK_MAXN 0
#   endif
# endif
# if (1 < INTERNAL_REGLOCK_MAXN)
#   if !defined(LIBXSMM_CACHE_MAXSIZE) && (8 > INTERNAL_REGLOCK_MAXN)
#     define LIBXSMM_CACHE_MAXSIZE LIBXSMM_CAPACITY_CACHE
#   endif
#   if !defined(LIBXSMM_REGLOCK)
#     define LIBXSMM_REGLOCK LIBXSMM_LOCK_DEFAULT
#   endif
#   if !defined(LIBXSMM_CLEANUP_NTRY)
#     define LIBXSMM_CLEANUP_NTRY 7
#   endif
#   if LIBXSMM_LOCK_TYPE_ISPOD(LIBXSMM_REGLOCK)
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE internal_reglocktype {
  char pad[LIBXSMM_CACHELINE];
  LIBXSMM_LOCK_TYPE(LIBXSMM_REGLOCK) state;
} internal_reglocktype;
#   else
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE internal_reglocktype {
  LIBXSMM_LOCK_TYPE(LIBXSMM_REGLOCK) state;
} internal_reglocktype;
#   endif
LIBXSMM_APIVAR_DEFINE(internal_reglocktype internal_reglock[INTERNAL_REGLOCK_MAXN]);
# else /* RW-lock */
#   if !defined(LIBXSMM_CACHE_MAXSIZE)
#     define LIBXSMM_CACHE_MAXSIZE LIBXSMM_CAPACITY_CACHE
#   endif
#   if !defined(LIBXSMM_REGLOCK)
#     if defined(LIBXSMM_UNIFY_LOCKS)
#       define LIBXSMM_REGLOCK LIBXSMM_LOCK
#     elif defined(_MSC_VER)
#       define LIBXSMM_REGLOCK LIBXSMM_LOCK_MUTEX
#     elif 0
#       define LIBXSMM_REGLOCK LIBXSMM_LOCK_RWLOCK
#     else
#       define LIBXSMM_REGLOCK LIBXSMM_LOCK_DEFAULT
#     endif
#   endif
LIBXSMM_APIVAR_DEFINE(LIBXSMM_LOCK_TYPE(LIBXSMM_REGLOCK)* internal_reglock_ptr);
# endif
#elif !defined(LIBXSMM_CACHE_MAXSIZE)
# define LIBXSMM_CACHE_MAXSIZE LIBXSMM_CAPACITY_CACHE
#endif
#if defined(LIBXSMM_UNPACKED) /* CCE/Classic */
# define LIBXSMM_CACHE_STRIDE LIBXSMM_MAX(sizeof(libxsmm_descriptor), LIBXSMM_DESCRIPTOR_MAXSIZE)
#else
# define LIBXSMM_CACHE_STRIDE LIBXSMM_DESCRIPTOR_MAXSIZE
#endif

#if defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
# define INTERNAL_FIND_CODE_CACHE_GROW(RESULT_INDEX, CACHE_SIZE) \
    RESULT_INDEX = CACHE_SIZE; CACHE_SIZE = (unsigned char)(0 != (CACHE_SIZE) ? ((CACHE_SIZE) << 1) : 1)
# define INTERNAL_FIND_CODE_CACHE_EVICT(RESULT_INDEX, CACHE_SIZE, CACHE_HIT) \
    RESULT_INDEX = (unsigned char)LIBXSMM_MOD2((CACHE_HIT) + ((CACHE_SIZE) - 1), CACHE_SIZE)
#endif

#if (0 == LIBXSMM_SYNC)
# define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) {
# define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) }
#else
# if defined(LIBXSMM_REGLOCK_TRY)
#   define INTERNAL_REGLOCK_TRY(DIFF, CODE) \
    if (1 != internal_reglock_count) { /* (re-)try and get (meanwhile) generated code */ \
      LIBXSMM_ASSERT(NULL != internal_registry); /* engine is not shut down */ \
      continue; \
    } \
    else { /* exit dispatch and let client fall back */ \
      DIFF = 0; CODE = 0; break; \
    }
# else
#   define INTERNAL_REGLOCK_TRY(DIFF, CODE) \
      LIBXSMM_ASSERT(NULL != internal_registry); /* engine is not shut down */ \
      continue
# endif
# if (1 < INTERNAL_REGLOCK_MAXN)
#   define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) { \
      const unsigned int LOCKINDEX = (0 != internal_reglock_count ? LIBXSMM_MOD2(INDEX, internal_reglock_count) : 0); \
      if (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_REGLOCK) != LIBXSMM_LOCK_TRYLOCK(LIBXSMM_REGLOCK, &internal_reglock[LOCKINDEX].state)) { \
        INTERNAL_REGLOCK_TRY(DIFF, CODE); \
      }
#   define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXSMM_LOCK_RELEASE(LIBXSMM_REGLOCK, &internal_reglock[LOCKINDEX].state); }
# else /* RW-lock */
#   define INTERNAL_FIND_CODE_LOCK(LOCKINDEX, INDEX, DIFF, CODE) { \
      if (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_REGLOCK) != LIBXSMM_LOCK_TRYLOCK(LIBXSMM_REGLOCK, internal_reglock_ptr)) { \
        INTERNAL_REGLOCK_TRY(DIFF, CODE); \
      }
#   define INTERNAL_FIND_CODE_UNLOCK(LOCKINDEX) LIBXSMM_LOCK_RELEASE(LIBXSMM_REGLOCK, internal_reglock_ptr); }
# endif
#endif


LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE internal_statistic_type {
  unsigned int ntry, ncol, njit, nsta;
} internal_statistic_type;

#if defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE internal_cache_entry_type {
  libxsmm_descriptor keys[LIBXSMM_CACHE_MAXSIZE];
  libxsmm_code_pointer code[LIBXSMM_CACHE_MAXSIZE];
  unsigned int id; /* to invalidate */
  unsigned char size, hit;
} internal_cache_entry_type;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE internal_cache_type {
# if defined(LIBXSMM_CACHE_PAD)
  char pad[LIBXSMM_UP2(sizeof(internal_cache_entry_type),LIBXSMM_CACHELINE)];
# endif
  internal_cache_entry_type entry;
} internal_cache_type;

# if defined(LIBXSMM_NTHREADS_USE)
LIBXSMM_APIVAR_DEFINE(internal_cache_type* internal_cache_buffer);
# endif
LIBXSMM_APIVAR_DEFINE(int internal_cache_size);
#endif /*defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))*/

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE internal_regkey_type {
#if defined(LIBXSMM_REGKEY_PAD)
  char pad[LIBXSMM_UP2(sizeof(libxsmm_descriptor), LIBXSMM_CACHELINE)];
#endif
  libxsmm_descriptor entry;
} internal_regkey_type;

/** Determines the try-lock property (1<N: disabled, N=1: enabled [N=0: disabled in case of RW-lock]). */
LIBXSMM_APIVAR_DEFINE(int internal_reglock_count);
LIBXSMM_APIVAR_DEFINE(size_t internal_registry_nbytes);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_registry_nleaks);
LIBXSMM_APIVAR_DEFINE(internal_regkey_type* internal_registry_keys);
LIBXSMM_APIVAR_DEFINE(libxsmm_code_pointer* internal_registry);
LIBXSMM_APIVAR_DEFINE(internal_statistic_type internal_statistic[2/*DP/SP*/][4/*sml/med/big/xxx*/]);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_statistic_sml);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_statistic_med);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_statistic_mnk);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_statistic_num_gemv);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_statistic_num_meltw);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_statistic_num_user);
LIBXSMM_APIVAR_DEFINE(int internal_gemm_auto_prefetch_locked);
LIBXSMM_APIVAR_DEFINE(const char* internal_build_state);
/** Time stamp (startup time of library). */
LIBXSMM_APIVAR_DEFINE(libxsmm_timer_tickint internal_timer_start);
LIBXSMM_APIVAR_DEFINE(libxsmm_cpuid_info internal_cpuid_info);

#if defined(_WIN32)
# define INTERNAL_SINGLETON_HANDLE HANDLE
# define INTERNAL_SINGLETON(HANDLE) (NULL != (HANDLE))
#else
# define INTERNAL_SINGLETON_HANDLE int
# define INTERNAL_SINGLETON(HANDLE) (0 <= (HANDLE) && '\0' != *internal_singleton_fname)
LIBXSMM_APIVAR_DEFINE(char internal_singleton_fname[64]);
#endif
LIBXSMM_APIVAR_DEFINE(INTERNAL_SINGLETON_HANDLE internal_singleton_handle);

/* definition of corresponding variables */
LIBXSMM_APIVAR_PRIVATE_DEF(libxsmm_malloc_function libxsmm_default_malloc_fn);
LIBXSMM_APIVAR_PRIVATE_DEF(libxsmm_malloc_function libxsmm_scratch_malloc_fn);
LIBXSMM_APIVAR_PRIVATE_DEF(libxsmm_free_function libxsmm_default_free_fn);
LIBXSMM_APIVAR_PRIVATE_DEF(libxsmm_free_function libxsmm_scratch_free_fn);
LIBXSMM_APIVAR_PRIVATE_DEF(const void* libxsmm_default_allocator_context);
LIBXSMM_APIVAR_PRIVATE_DEF(const void* libxsmm_scratch_allocator_context);
LIBXSMM_APIVAR_PRIVATE_DEF(unsigned int libxsmm_scratch_pools);
LIBXSMM_APIVAR_PRIVATE_DEF(double libxsmm_scratch_scale);
LIBXSMM_APIVAR_PRIVATE_DEF(double libxsmm_timer_scale);
LIBXSMM_APIVAR_PRIVATE_DEF(unsigned int libxsmm_statistic_num_spmdm);
LIBXSMM_APIVAR_PRIVATE_DEF(unsigned int libxsmm_thread_count);
/* definition of corresponding variables */
LIBXSMM_APIVAR_PUBLIC_DEF(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK) libxsmm_lock_global);
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_nosync);

#if (0 != LIBXSMM_SYNC)
LIBXSMM_APIVAR_PRIVATE_DEF(LIBXSMM_TLS_TYPE libxsmm_tlskey);
#endif


LIBXSMM_API_INTERN void* libxsmm_memalign_internal(size_t alignment, size_t size)
{
  void* result;
  LIBXSMM_ASSERT(LIBXSMM_ISPOT(alignment));
#if defined(LIBXSMM_MALLOC_HOOK_INTRINSIC)
  result = _mm_malloc(size, alignment);
#elif (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD))) /* GLIBC */
  result = __libc_memalign(alignment, size);
#elif defined(LIBXSMM_BUILD) && ( /*C11*/ \
  defined(__STDC_VERSION__) && (201112L <= __STDC_VERSION__))
  result = aligned_alloc(alignment, LIBXSMM_UP2(size, alignment));
#elif (defined(_WIN32) || defined(__CYGWIN__))
  LIBXSMM_UNUSED(alignment);
  result = malloc(size);
#elif defined(NDEBUG)
  posix_memalign(&result, alignment, size);
#else
  if (0 != posix_memalign(&result, alignment, size)) result = NULL;
#endif
  return result;
}


LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_WEAK void* __real_memalign(size_t alignment, size_t size)
{
  void* result;
#if defined(LIBXSMM_MALLOC_HOOK_DYNAMIC)
  if (NULL != libxsmm_malloc_fn.memalign.ptr) {
    result = libxsmm_malloc_fn.memalign.ptr(alignment, size);
  }
  else
#endif
  result = libxsmm_memalign_internal(alignment, size);
  return result;
}


LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_WEAK void* __real_malloc(size_t size)
{
  void* result;
#if defined(LIBXSMM_MALLOC_HOOK_ALIGN)
  result = __real_memalign(libxsmm_alignment(size, 0/*auto*/), size);
#else
# if defined(LIBXSMM_MALLOC_HOOK_DYNAMIC)
  if (NULL != libxsmm_malloc_fn.malloc.ptr) {
    LIBXSMM_ASSERT(malloc != libxsmm_malloc_fn.malloc.ptr);
    result = libxsmm_malloc_fn.malloc.ptr(size);
  }
  else
# endif
# if defined(LIBXSMM_MALLOC_HOOK_INTRINSIC)
  result = _mm_malloc(size, libxsmm_alignment(size, 0/*auto*/));
# elif (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD))) /* GLIBC */
  result = __libc_malloc(size);
# else
  result = malloc(size);
# endif
#endif
  return result;
}


#if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_WEAK void* __real_calloc(size_t num, size_t size)
{
  void* result;
#if defined(LIBXSMM_MALLOC_HOOK_DYNAMIC)
  if (NULL != libxsmm_malloc_fn.calloc.ptr) {
    LIBXSMM_ASSERT(calloc != libxsmm_malloc_fn.calloc.ptr);
    result = libxsmm_malloc_fn.calloc.ptr(num, size);
  }
  else
#endif
#if defined(LIBXSMM_MALLOC_HOOK_INTRINSIC)
  { const size_t num_size = num * size;
    result = _mm_malloc(num_size, libxsmm_alignment(num_size, 0/*auto*/));
    if (NULL != result) memset(result, 0, num_size);
  }
#elif (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD))) /* GLIBC */
  result = __libc_calloc(num, size);
#else
  result = calloc(num, size);
#endif
  return result;
}
#endif


#if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_WEAK void* __real_realloc(void* ptr, size_t size)
{
  void* result;
#if defined(LIBXSMM_MALLOC_HOOK_DYNAMIC)
  if (NULL != libxsmm_malloc_fn.realloc.ptr) {
    LIBXSMM_ASSERT(realloc != libxsmm_malloc_fn.realloc.ptr);
    result = libxsmm_malloc_fn.realloc.ptr(ptr, size);
  }
  else
#endif
#if (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD))) /* GLIBC */
  result = __libc_realloc(ptr, size);
#else
  result = realloc(ptr, size);
#endif
  return result;
}
#endif


LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_WEAK void __real_free(void* ptr)
{
  if (NULL != ptr) {
#if defined(LIBXSMM_MALLOC_HOOK_DYNAMIC)
    if (NULL != libxsmm_malloc_fn.free.ptr) {
      LIBXSMM_ASSERT(free != libxsmm_malloc_fn.free.ptr);
      libxsmm_malloc_fn.free.ptr(ptr);
    }
    else
#endif
#if defined(LIBXSMM_MALLOC_HOOK_INTRINSIC)
    { static int recursive = 0;
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&recursive, 1, LIBXSMM_ATOMIC_RELAXED)) _mm_free(ptr);
      else {
# if (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD))) /* GLIBC */
        __libc_free(ptr);
# else
        free(ptr);
# endif
      }
      LIBXSMM_ATOMIC_SUB_FETCH(&recursive, 1, LIBXSMM_ATOMIC_RELAXED);
    }
#elif (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD))) /* GLIBC */
    __libc_free(ptr);
#else
    free(ptr);
#endif
  }
}


LIBXSMM_API_INLINE void internal_update_mmstatistic(const libxsmm_gemm_descriptor* desc,
  unsigned int ntry, unsigned int ncol, unsigned int njit, unsigned int nsta)
{
  LIBXSMM_ASSERT(NULL != desc);
  if (1 < desc->m && 1 < desc->n) { /* only record matrix-matrix multiplication */
    const unsigned long long kernel_size = LIBXSMM_MNK_SIZE(desc->m, desc->n, desc->k);
    const int idx = (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_OUT(desc->datatype) ? 0 : 1);
    int bucket;
    if (LIBXSMM_MNK_SIZE(internal_statistic_sml, internal_statistic_sml, internal_statistic_sml) >= kernel_size) {
      bucket = 0;
    }
    else if (LIBXSMM_MNK_SIZE(internal_statistic_med, internal_statistic_med, internal_statistic_med) >= kernel_size) {
      bucket = 1;
    }
    else if (LIBXSMM_MNK_SIZE(internal_statistic_mnk, internal_statistic_mnk, internal_statistic_mnk) >= kernel_size) {
      bucket = 2;
    }
    else { /*huge*/
      bucket = 3;
    }
    if (0 != ncol) ncol/*dummy assignment*/ = LIBXSMM_ATOMIC_ADD_FETCH(&internal_statistic[idx][bucket].ncol, ncol, LIBXSMM_ATOMIC_RELAXED);
    if (0 != ntry) ntry/*dummy assignment*/ = LIBXSMM_ATOMIC_ADD_FETCH(&internal_statistic[idx][bucket].ntry, ntry, LIBXSMM_ATOMIC_RELAXED);
    /* the following counters are not manipulated concurrently (no need for atomic increment) */
    if (0 != njit) internal_statistic[idx][bucket].njit += njit;
    if (0 != nsta) internal_statistic[idx][bucket].nsta += nsta;
  }
}


LIBXSMM_API_INLINE unsigned int internal_print_number(unsigned int n, char default_unit, char* unit)
{
  unsigned int number = n;
  LIBXSMM_ASSERT(NULL != unit);
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


LIBXSMM_API_INLINE unsigned int internal_print_statistic(FILE* ostream,
  const char* target_arch, int precision, unsigned int linebreaks, unsigned int indent)
{
  const internal_statistic_type statistic_sml = internal_statistic[precision][0/*SML*/];
  const internal_statistic_type statistic_med = internal_statistic[precision][1/*MED*/];
  const internal_statistic_type statistic_big = internal_statistic[precision][2/*BIG*/];
  const internal_statistic_type statistic_xxx = internal_statistic[precision][3/*XXX*/];
  int printed = 0;
  LIBXSMM_ASSERT(NULL != ostream && (0 <= precision && precision < 2));
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
      if (NULL != target_arch && '\0' != *target_arch) {
        assert(strlen(target_arch) < sizeof(title)); /* !LIBXSMM_ASSERT */
        for (n = 0; 0 != target_arch[n] /*avoid code-gen. issue with some clang versions: && n < sizeof(title)*/; ++n) {
          const char c = target_arch[n];
          title[n] = (char)(('a' <= c && c <= 'z') ? (c - 32) : c); /* toupper */
        }
        LIBXSMM_SNPRINTF(title + n, sizeof(title) - n, "/%s", 0 == precision ? "DP" : "SP");
      }
      else {
        LIBXSMM_SNPRINTF(title, sizeof(title), "%s", 0 == precision ? "DP" : "SP");
      }
      for (n = 0; n < linebreaks; ++n) fprintf(ostream, "\n");
    }
    fprintf(ostream, "%*s%-8s %6s %6s %6s %6s\n", (int)indent, "", title, "TRY", "JIT", "STA", "COL");
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


#if !(defined(_WIN32) || defined(__CYGWIN__))
LIBXSMM_API_INLINE unsigned int internal_statistic_ntry(int precision)
{
  return internal_statistic[precision][0/*SML*/].ntry + internal_statistic[precision][1/*MED*/].ntry
       + internal_statistic[precision][2/*BIG*/].ntry + internal_statistic[precision][3/*XXX*/].ntry;
}
#endif


#if !defined(_WIN32)
LIBXSMM_API_INLINE void internal_register_static_code(
  libxsmm_gemm_precision precision, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_xmmfunction xgemm, libxsmm_code_pointer* registry)
{
  const libxsmm_blasint lda = m, ldb = k, ldc = m;
  /*const*/ int precondition = LIBXSMM_GEMM_NO_BYPASS_DIMS(m, n, k) && LIBXSMM_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc);
  if (precondition) {
    const size_t size = (LIBXSMM_HASH_SIZE) - sizeof(libxsmm_descriptor_kind);
    libxsmm_descriptor_blob blob;
    const libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_dinit(&blob, precision,
      m, n, k, lda, ldb, ldc, LIBXSMM_ALPHA, LIBXSMM_BETA, LIBXSMM_FLAGS, INTERNAL_PREFETCH);
    unsigned int i = LIBXSMM_MOD2(
      libxsmm_crc32(LIBXSMM_HASH_SEED, desc, LIBXSMM_MIN(sizeof(libxsmm_gemm_descriptor), size)),
      LIBXSMM_CAPACITY_REGISTRY);
    libxsmm_code_pointer* dst_entry = registry + i;
#if !defined(NDEBUG)
    libxsmm_code_pointer code; code.xgemm = xgemm;
    LIBXSMM_ASSERT(NULL != code.ptr_const && NULL != registry);
    LIBXSMM_ASSERT(0 == (LIBXSMM_CODE_STATIC & code.uval));
#endif
    if (NULL != dst_entry->ptr_const) { /* collision */
      const unsigned int i0 = i;
      do { /* continue to linearly search for an available slot */
        i = LIBXSMM_MOD2(i + 1, LIBXSMM_CAPACITY_REGISTRY);
        if (NULL == registry[i].ptr_const) break;
      } while (i != i0);
#if defined(LIBXSMM_HASH_COLLISION) /* mark entry as a collision */
      dst_entry->uval |= LIBXSMM_HASH_COLLISION;
#endif
      dst_entry = registry + i; /* update destination */
      internal_update_mmstatistic(desc, 0, 1/*collision*/, 0, 0);
      /* out of capacity (no registry slot available) */
      LIBXSMM_ASSERT(NULL == dst_entry->ptr_const || i == i0);
    }
    if (NULL == dst_entry->ptr_const) { /* registry not exhausted */
      internal_registry_keys[i].entry.kind = LIBXSMM_KERNEL_KIND_MATMUL;
      LIBXSMM_ASSIGN127(&internal_registry_keys[i].entry.gemm.desc, desc);
      dst_entry->xgemm = xgemm;
      /* mark current entry as static code (non-JIT) */
      dst_entry->uval |= LIBXSMM_CODE_STATIC;
    }
    internal_update_mmstatistic(desc, 1/*try*/, 0, 0, 0);
  }
}
#endif


LIBXSMM_API_INTERN void internal_release_scratch(void);
LIBXSMM_API_INTERN void internal_release_scratch(void)
{
  libxsmm_xrelease_scratch(NULL/*lock*/);
  /* release global services */
  libxsmm_memory_finalize();
  libxsmm_hash_finalize();
  libxsmm_malloc_finalize();
}


/* Caution: cannot be used multiple times in a single expression! */
LIBXSMM_API_INTERN size_t libxsmm_format_value(char buffer[32], int buffer_size, size_t nbytes, const char scale[], const char* unit, int base)
{
  const int len = (NULL != scale ? ((int)strlen(scale)) : 0);
  const int m = LIBXSMM_INTRINSICS_BITSCANBWD64(nbytes) / base, n = LIBXSMM_MIN(m, len);
  int i;
  buffer[0] = 0; /* clear */
  LIBXSMM_ASSERT(NULL != unit && 0 <= base);
  for (i = 0; i < n; ++i) nbytes >>= base;
  LIBXSMM_SNPRINTF(buffer, buffer_size, "%i %c%s",
    (int)nbytes, 0 < n ? scale[n-1] : *unit, 0 < n ? unit : "");
  return nbytes;
}


LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_NO_TRACE void internal_dump(FILE* ostream, int urgent);
LIBXSMM_API_INTERN void internal_dump(FILE* ostream, int urgent)
{
  char *const env_dump_build = getenv("LIBXSMM_DUMP_BUILD");
  char *const env_dump_files = (NULL != getenv("LIBXSMM_DUMP_FILES")
    ? getenv("LIBXSMM_DUMP_FILES")
    : getenv("LIBXSMM_DUMP_FILE"));
  LIBXSMM_ASSERT_MSG(INTERNAL_SINGLETON(internal_singleton_handle), "Invalid handle");
  /* determine whether this instance is unique or not */
  if (NULL != env_dump_files && '\0' != *env_dump_files && 0 == urgent) { /* dump per-node info */
    const char* filename = strtok(env_dump_files, INTERNAL_DELIMS);
    char buffer[1024];
    for (; NULL != filename; filename = strtok(NULL, INTERNAL_DELIMS)) {
      FILE* file = fopen(filename, "r");
      if (NULL != file) buffer[0] = '\0';
      else { /* parse keywords */
        const int seconds = atoi(filename);
        if (0 == seconds) {
          const char *const pid = strstr(filename, "PID");
          if (NULL != pid) { /* PID-keyword is present */
            int n = (int)(pid - filename);
            n = LIBXSMM_SNPRINTF(buffer, sizeof(buffer), "%.*s%u%s", n, filename, libxsmm_get_pid(), filename + n + 3);
            if (0 < n && (int)sizeof(buffer) > n) {
              file = fopen(buffer, "r");
              filename = buffer;
            }
          }
        }
        else {
          fprintf(stderr, "LIBXSMM INFO: PID=%u\n", libxsmm_get_pid());
          if (0 < seconds) {
#if defined(_WIN32)
            Sleep((DWORD)(1000 * seconds));
#else
            LIBXSMM_EXPECT(EXIT_SUCCESS, sleep(seconds));
#endif
          }
          else for(;;) LIBXSMM_SYNC_YIELD;
        }
      }
      if (NULL != file) {
        int c = fgetc(file);
        fprintf(ostream, "\n\nLIBXSMM_DUMP_FILE: %s\n", filename);
        /* coverity[tainted_data] */
        while (EOF != c) {
          fputc(c, stdout);
          c = fgetc(file);
        }
        fputc('\n', stdout);
        fclose(file);
      }
    }
  }
  if  (NULL != internal_build_state /* dump build state */
    && NULL != env_dump_build && '\0' != *env_dump_build)
  {
    const int dump_build = atoi(env_dump_build);
    if (0 == urgent ? (0 < dump_build) : (0 > dump_build)) {
      fprintf(ostream, "\n\nBUILD_DATE=%i\n", LIBXSMM_CONFIG_BUILD_DATE);
      fprintf(ostream, "%s\n", internal_build_state);
    }
  }
}


LIBXSMM_API_INTERN void internal_finalize(void);
LIBXSMM_API_INTERN void internal_finalize(void)
{
  libxsmm_finalize();
  LIBXSMM_STDIO_ACQUIRE(); /* synchronize I/O */
  if (0 != libxsmm_verbosity) { /* print statistic on termination */
    const char *const env_target_hidden = getenv("LIBXSMM_TARGET_HIDDEN");
    const char *const target_arch = (NULL == env_target_hidden || 0 == atoi(env_target_hidden))
      ? libxsmm_cpuid_name(libxsmm_target_archid) : NULL/*hidden*/;
    fprintf(stderr, "\nLIBXSMM_VERSION: %s%s%s (%i)", LIBXSMM_BRANCH,
      0 != *(LIBXSMM_BRANCH) ? "-" : "", 0 != *(LIBXSMM_VERSION) ? (LIBXSMM_VERSION) : "unconfigured",
      LIBXSMM_VERSION4(LIBXSMM_VERSION_MAJOR, LIBXSMM_VERSION_MINOR, LIBXSMM_VERSION_UPDATE, LIBXSMM_VERSION_PATCH));
    if (LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity) {
      unsigned int linebreak = (0 == internal_print_statistic(stderr, target_arch, 1/*SP*/, 1, 0)) ? 1 : 0;
      const int high_verbosity = (LIBXSMM_VERBOSITY_HIGH <= libxsmm_verbosity || 0 > libxsmm_verbosity);
      char number_format_buffer[32];
      libxsmm_scratch_info scratch_info;
#if defined(LIBXSMM_PLATFORM_X86)
      libxsmm_cpuid_info info;
      libxsmm_cpuid_x86(&info);
      if ((LIBXSMM_VERBOSITY_HIGH < libxsmm_verbosity || 0 > libxsmm_verbosity) &&
        0 == internal_cpuid_info.has_context && 0 != info.has_context)
      {
        fprintf(stderr, "\nLIBXSMM: CPU features have been promoted.");
      }
#endif
      if (0 == internal_print_statistic(stderr, target_arch, 0/*DP*/, linebreak, 0) && 0 != linebreak && NULL != target_arch) {
        fprintf(stderr, "\nLIBXSMM_TARGET: %s\n", target_arch);
      }
      if (0 != libxsmm_format_value(number_format_buffer, sizeof(number_format_buffer),
#if defined(LIBXSMM_NTHREADS_USE) && defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
        sizeof(internal_cache_type) * (LIBXSMM_NTHREADS_MAX) +
#endif
        (sizeof(internal_regkey_type) + sizeof(libxsmm_code_pointer)) * (LIBXSMM_CAPACITY_REGISTRY),
        "KM", "B", 10))
      {
        fprintf(stderr, "Registry and code: %s", number_format_buffer);
        if (0 != libxsmm_format_value(number_format_buffer, sizeof(number_format_buffer), internal_registry_nbytes, "KM", "B", 10)) {
          fprintf(stderr, " + %s", number_format_buffer);
        }
        if (0 != high_verbosity) {
          unsigned int ngemms = 0;
          int i; for (i = 0; i < 4; ++i) {
            ngemms += internal_statistic[0/*DP*/][i].nsta + internal_statistic[1/*SP*/][i].nsta;
            ngemms += internal_statistic[0/*DP*/][i].njit + internal_statistic[1/*SP*/][i].njit;
          }
          if (0 != ngemms || 0 != internal_statistic_num_gemv
            || 0 != internal_statistic_num_meltw
            || 0 != libxsmm_statistic_num_spmdm
            || 0 != internal_statistic_num_user
            || 0 != internal_registry_nleaks)
          {
            const char sep[] = " ", *s = "";
            fprintf(stderr, " (");
            if (0 != ngemms) { fprintf(stderr, "gemm=%u", ngemms); s = sep; }
            if (0 != internal_statistic_num_gemv) { fprintf(stderr, "%sgemv=%u", s, internal_statistic_num_gemv); s = sep; }
            if (0 != internal_statistic_num_meltw) { fprintf(stderr, "%smeltw=%u", s, internal_statistic_num_meltw); s = sep; }
            if (0 != libxsmm_statistic_num_spmdm) { fprintf(stderr, "%sspmdm=%u", s, libxsmm_statistic_num_spmdm); s = sep; }
            if (0 != internal_statistic_num_user) { fprintf(stderr, "%suser=%u", s, internal_statistic_num_user); s = sep; }
            if (0 != internal_registry_nleaks) { fprintf(stderr, "%snleaks=%u", s, internal_registry_nleaks); s = sep; }
            fprintf(stderr, ")");
          }
        }
        fprintf(stderr, "\n");
      }
      if (EXIT_SUCCESS == libxsmm_get_scratch_info(&scratch_info)) {
        if (0 != scratch_info.size &&
          0 != libxsmm_format_value(number_format_buffer, sizeof(number_format_buffer), scratch_info.size, "KM", "B", 10))
        {
          fprintf(stderr, "Scratch: %s", number_format_buffer);
          if (0 != high_verbosity) {
            fprintf(stderr, " (mallocs=%lu, pools=%u)\n", (unsigned long int)scratch_info.nmallocs, scratch_info.npools);
          }
          else {
            fprintf(stderr, "\n");
          }
        }
        if (0 != scratch_info.internal && 0 != high_verbosity &&
          libxsmm_format_value(number_format_buffer, sizeof(number_format_buffer), scratch_info.internal, "KM", "B", 10))
        {
          fprintf(stderr, "Private: %s\n", number_format_buffer);
        }
      }
      if (LIBXSMM_VERBOSITY_HIGH < libxsmm_verbosity || 0 > libxsmm_verbosity) {
        fprintf(stderr, "Uptime: %f s", libxsmm_timer_duration(internal_timer_start, libxsmm_timer_tick()));
        if (1 < libxsmm_thread_count && INT_MAX == libxsmm_verbosity) {
          fprintf(stderr, " (nthreads=%u)", libxsmm_thread_count);
        }
        fprintf(stderr, "\n");
      }
    }
    else {
      fprintf(stderr, "\nLIBXSMM_TARGET: %s\n", target_arch);
    }
  }
  /* release scratch memory pool */
  if (EXIT_SUCCESS != atexit(internal_release_scratch) && 0 != libxsmm_verbosity) {
    fprintf(stderr, "LIBXSMM ERROR: failed to perform final cleanup!\n");
  }
  /* determine whether this instance is unique or not */
  if (INTERNAL_SINGLETON(internal_singleton_handle)) {
    internal_dump(stdout, 0/*urgent*/);
    /* cleanup singleton */
#if defined(_WIN32)
    ReleaseMutex(internal_singleton_handle);
    CloseHandle(internal_singleton_handle);
#else
    unlink(internal_singleton_fname);
    close(internal_singleton_handle);
#endif
  }
  LIBXSMM_STDIO_RELEASE(); /* synchronize I/O */
#if (0 != LIBXSMM_SYNC)
  { /* release locks */
# if (1 < INTERNAL_REGLOCK_MAXN)
    int i; for (i = 0; i < internal_reglock_count; ++i) LIBXSMM_LOCK_DESTROY(LIBXSMM_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXSMM_UNIFY_LOCKS)
    LIBXSMM_LOCK_DESTROY(LIBXSMM_REGLOCK, internal_reglock_ptr);
# endif
    LIBXSMM_LOCK_DESTROY(LIBXSMM_LOCK, &libxsmm_lock_global);
  }
#endif
}


#if defined(LIBXSMM_INTERCEPT_DYNAMIC)
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void _gfortran_stop_string(const char* /*message*/, int /*len*/, int /*quiet*/);
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void _gfortran_stop_string(const char* message, int len, int quiet)
{ /* STOP termination handler for GNU Fortran runtime */
  static int once = 0;
  if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&once, 1, LIBXSMM_ATOMIC_RELAXED)) {
    union { const void* dlsym; void (*ptr)(const char*, int, int); } stop;
    dlerror(); /* clear an eventual error status */
    stop.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "_gfortran_stop_string");
    if (NULL != stop.dlsym) {
      stop.ptr(message, len, quiet);
    }
    else exit(EXIT_SUCCESS); /* statically linked runtime */
  }
}

LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void for_stop_core(const char* /*message*/, int /*len*/);
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void for_stop_core(const char* message, int len)
{ /* STOP termination handler for Intel Fortran runtime */
  static int once = 0;
  if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&once, 1, LIBXSMM_ATOMIC_RELAXED)) {
    union { const void* dlsym; void (*ptr)(const char*, int); } stop;
    dlerror(); /* clear an eventual error status */
    stop.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "for_stop_core");
    if (NULL != stop.dlsym) {
      stop.ptr(message, len);
    }
    else exit(EXIT_SUCCESS); /* statically linked runtime */
  }
}

LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void for_stop_core_quiet(void);
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void for_stop_core_quiet(void)
{ /* STOP termination handler for Intel Fortran runtime */
  static int once = 0;
  if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&once, 1, LIBXSMM_ATOMIC_RELAXED)) {
    union { const void* dlsym; void (*ptr)(void); } stop;
    dlerror(); /* clear an eventual error status */
    stop.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "for_stop_core_quiet");
    if (NULL != stop.dlsym) {
      stop.ptr();
    }
    else exit(EXIT_SUCCESS); /* statically linked runtime */
  }
}
#endif


LIBXSMM_API_INTERN size_t internal_strlen(const char* /*cstr*/, size_t /*maxlen*/);
LIBXSMM_API_INTERN size_t internal_strlen(const char* cstr, size_t maxlen)
{
  size_t result = 0;
  if (NULL != cstr) {
    while (0 != cstr[result] && result < maxlen) ++result;
  }
  return result;
}


LIBXSMM_API_INTERN size_t internal_parse_nbytes(const char* /*nbytes*/, size_t /*ndefault*/, int* /*valid*/);
LIBXSMM_API_INTERN size_t internal_parse_nbytes(const char* nbytes, size_t ndefault, int* valid)
{
  size_t result = ndefault;
  if (NULL != nbytes && '\0' != *nbytes) {
    size_t u = internal_strlen(nbytes, 32) - 1;
    const char units[] = "kmgKMG", *const unit = strchr(units, nbytes[u]);
    char* end = NULL;
    /* take parsed value with increased type-width */
    const long long int ibytes = strtol(nbytes, &end, 10);
    if (NULL != end && ( /* no obvious error */
      /* must match allowed set of units */
      (NULL != unit && *unit == *end) ||
      /* value is given without unit */
      (NULL == unit && '\0' == *end)))
    {
      result = (size_t)ibytes;
      if ((size_t)LIBXSMM_UNLIMITED != result) {
        u = (NULL != unit ? ((unit - units) % 3) : 3);
        if (u < 3) {
          result <<= (u + 1) * 10;
        }
      }
      if (NULL != valid) *valid = 1;
    }
    else if (NULL != valid) *valid = 0;
  }
  else if (NULL != valid) {
    *valid = 0;
  }
  return result;
}


LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_NO_TRACE void internal_init(void);
LIBXSMM_API_INTERN void internal_init(void)
{
  int i;
#if (0 != LIBXSMM_SYNC) /* setup the locks in a thread-safe fashion */
  LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK, &libxsmm_lock_global);
# if (1 < INTERNAL_REGLOCK_MAXN)
  for (i = 0; i < internal_reglock_count; ++i) LIBXSMM_LOCK_ACQUIRE(LIBXSMM_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXSMM_UNIFY_LOCKS)
  LIBXSMM_LOCK_ACQUIRE(LIBXSMM_REGLOCK, internal_reglock_ptr);
# endif
#endif
  if (NULL == internal_registry) { /* double-check after acquiring the lock(s) */
#if defined(LIBXSMM_INTERCEPT_DYNAMIC) && defined(LIBXSMM_AUTOPIN)
    /* clear error status (dummy condition: it does not matter if MPI_Init or MPI_Abort) */
    const char *const dlsymname = (NULL == dlerror() ? "MPI_Init" : "MPI_Abort");
    const void *const dlsymbol = dlsym(LIBXSMM_RTLD_NEXT, dlsymname);
    const void *const dlmpi = (NULL == dlerror() ? dlsymbol : NULL);
#endif
    const char *const env_verbose = getenv("LIBXSMM_VERBOSE");
    void* new_registry = NULL, * new_keys = NULL;
#if defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
# if defined(LIBXSMM_NTHREADS_USE)
    void* new_cache = NULL;
# endif
    const char *const env_cache = getenv("LIBXSMM_CACHE");
    if (NULL != env_cache && '\0' != *env_cache) {
      const int cache_size = atoi(env_cache), cache_size2 = LIBXSMM_UP2POT(cache_size);
      internal_cache_size = LIBXSMM_MIN(cache_size2, LIBXSMM_CACHE_MAXSIZE);
    }
    else {
      internal_cache_size = LIBXSMM_CACHE_MAXSIZE;
    }
#endif
    /* setup verbosity as early as possible since below code may rely on verbose output */
    if (NULL != env_verbose && '\0' != *env_verbose) {
      libxsmm_verbosity = atoi(env_verbose);
    }
#if !defined(NDEBUG)
    else {
      libxsmm_verbosity = INT_MAX; /* quiet -> verbose */
    }
#endif
#if (0 == LIBXSMM_JIT)
    if (2 > libxsmm_ninit && (LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity)) {
      fprintf(stderr, "LIBXSMM: JIT-code generation was disabled at compile-time.\n");
    }
#endif
#if defined(LIBXSMM_AUTOPIN)
# if defined(LIBXSMM_INTERCEPT_DYNAMIC)
    /* MPI: unwanted affinity can slow-down unrelated jobs (over-subscription), e.g., CP2K regtests */
    if (NULL == dlmpi)
# endif
    { /* setup some viable affinity if nothing else is present */
      const char *const gomp_cpu_affinity = getenv("GOMP_CPU_AFFINITY");
      const char *const kmp_affinity = getenv("KMP_AFFINITY");
      const char *const omp_proc_bind = getenv("OMP_PROC_BIND");
      if  ((NULL == gomp_cpu_affinity || 0 == *gomp_cpu_affinity)
        && (NULL == kmp_affinity || 0 == *kmp_affinity)
        && (NULL == omp_proc_bind || 0 == *omp_proc_bind))
      {
        static char affinity[] = "OMP_PROC_BIND=TRUE";
        LIBXSMM_EXPECT(EXIT_SUCCESS, LIBXSMM_PUTENV(affinity));
        if (LIBXSMM_VERBOSITY_HIGH < libxsmm_verbosity || 0 > libxsmm_verbosity) { /* library code is expected to be mute */
          fprintf(stderr, "LIBXSMM: prepared to pin threads.\n");
        }
      }
    }
# if defined(LIBXSMM_INTERCEPT_DYNAMIC) && 1
    else if (NULL == getenv("I_MPI_SHM_HEAP")) {
      static char shmheap[] = "I_MPI_SHM_HEAP=1";
      LIBXSMM_EXPECT(EXIT_SUCCESS, LIBXSMM_PUTENV(shmheap));
    }
# endif
#endif
#if !defined(_WIN32) && 0
    umask(S_IRUSR | S_IWUSR); /* setup default/secure file mask */
#endif
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
    { const char *const env = getenv("LIBXSMM_SCRATCH_POOLS");
      if (NULL == env || 0 == *env) {
        libxsmm_scratch_pools = LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS;
      }
      else {
        libxsmm_scratch_pools = LIBXSMM_CLMP(atoi(env), 0, LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS);
        /*libxsmm_scratch_pools_locked = 1;*/
      }
      LIBXSMM_ASSERT(libxsmm_scratch_pools <= LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS);
    }
    { const char *const env = getenv("LIBXSMM_SCRATCH_SCALE");
      if (NULL == env || 0 == *env) {
        libxsmm_scratch_scale = LIBXSMM_MALLOC_SCRATCH_SCALE;
      }
      else {
        libxsmm_scratch_scale = LIBXSMM_CLMP(atof(env), 1.0, 10.0);
        /*libxsmm_scratch_scale_locked = 1;*/
      }
      assert(1 <= libxsmm_scratch_scale); /* !LIBXSMM_ASSERT */
    }
    libxsmm_set_scratch_limit(internal_parse_nbytes(getenv("LIBXSMM_SCRATCH_LIMIT"), LIBXSMM_SCRATCH_DEFAULT, NULL/*valid*/));
#endif /*defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))*/
    { /* setup malloc-interception after internal allocations */
      const libxsmm_malloc_function null_malloc_fn = { 0 };
      const libxsmm_free_function null_free_fn = { 0 };
      char *const env_k = getenv("LIBXSMM_MALLOC"), *const env_t = getenv("LIBXSMM_MALLOC_LIMIT"), *end = NULL;
      const char* env_i = (NULL != env_t ? strtok(env_t, INTERNAL_DELIMS) : NULL);
      size_t malloc_lo = internal_parse_nbytes(env_i, LIBXSMM_MALLOC_LIMIT, NULL/*valid*/);
      size_t malloc_hi = (NULL != env_i ? internal_parse_nbytes(
        strtok(NULL, INTERNAL_DELIMS), LIBXSMM_SCRATCH_UNLIMITED, NULL/*valid*/) : LIBXSMM_SCRATCH_UNLIMITED);
      const int malloc_kind = ((NULL == env_k || 0 == *env_k) ? 0/*disabled*/ : ((int)strtol(env_k, &end, 10)));
      libxsmm_xset_default_allocator(NULL/*lock*/, NULL/*context*/, null_malloc_fn, null_free_fn);
      libxsmm_xset_scratch_allocator(NULL/*lock*/, NULL/*context*/, null_malloc_fn, null_free_fn);
      /* libxsmm_set_malloc implies libxsmm_malloc_init */
      if (NULL == end) {
        libxsmm_set_malloc(0, &malloc_lo, &malloc_hi);
      }
      else if ('\0' == *end) {
        libxsmm_set_malloc(malloc_kind, &malloc_lo, &malloc_hi);
      }
      else {
        int valid = 1;
        env_i = strtok(env_k, INTERNAL_DELIMS);
        malloc_lo = internal_parse_nbytes(env_i, LIBXSMM_MALLOC_LIMIT, &valid);
        env_i = (0 != valid ? strtok(NULL, INTERNAL_DELIMS) : NULL);
        malloc_hi = (NULL != env_i
          ? internal_parse_nbytes(env_i, LIBXSMM_SCRATCH_UNLIMITED, &valid)
          : LIBXSMM_SCRATCH_UNLIMITED);
        libxsmm_set_malloc(0 != valid ? 1 : 0, &malloc_lo, &malloc_hi);
      }
    }
#if defined(LIBXSMM_MAXTARGET)
    libxsmm_set_target_arch(LIBXSMM_STRINGIFY(LIBXSMM_MAXTARGET));
#else /* attempt to set libxsmm_target_archid per environment variable */
    libxsmm_set_target_arch(getenv("LIBXSMM_TARGET"));
#endif
    { const char *const env = getenv("LIBXSMM_SYNC");
      libxsmm_nosync = (NULL == env || 0 == *env) ? 0/*default*/ : atoi(env);
    }
    /* clear internal counters/statistic */
    for (i = 0; i < 4/*sml/med/big/xxx*/; ++i) {
      LIBXSMM_MEMZERO127(&internal_statistic[0/*DP*/][i]);
      LIBXSMM_MEMZERO127(&internal_statistic[1/*SP*/][i]);
    }
    internal_statistic_mnk = LIBXSMM_MAX_DIM;
    internal_statistic_sml = 13;
    internal_statistic_med = 23;
    LIBXSMM_ASSERT(LIBXSMM_ISPOT(LIBXSMM_CAPACITY_REGISTRY));
    libxsmm_hash_init(libxsmm_target_archid); /* used by debug memory allocation (checksum) */
    libxsmm_memory_init(libxsmm_target_archid);
    if (
#if defined(LIBXSMM_NTHREADS_USE) && defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
      (EXIT_SUCCESS == libxsmm_xmalloc(&new_cache, /* if internal_cache_size is zero, allocation must still happen (later control-flow too expensive) */
        sizeof(internal_cache_type) * (LIBXSMM_NTHREADS_MAX), LIBXSMM_CACHELINE/*alignment*/,
        LIBXSMM_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_cache) &&
#endif
      (EXIT_SUCCESS == libxsmm_xmalloc(&new_keys, (LIBXSMM_CAPACITY_REGISTRY) * sizeof(internal_regkey_type), 0/*auto-align*/,
        LIBXSMM_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_keys) &&
      (EXIT_SUCCESS == libxsmm_xmalloc(&new_registry, (LIBXSMM_CAPACITY_REGISTRY) * sizeof(libxsmm_code_pointer), 0/*auto-align*/,
        LIBXSMM_MALLOC_FLAG_PRIVATE, NULL/*extra*/, 0/*extra-size*/) && NULL != new_registry))
    {
#if defined(LIBXSMM_NTHREADS_USE) && defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
      LIBXSMM_ASSERT(NULL != new_cache); /* SA: suppress false positive */
      memset(new_cache, 0, (LIBXSMM_NTHREADS_MAX) * sizeof(internal_cache_type));
#endif
      libxsmm_xcopy_init(libxsmm_target_archid);
      libxsmm_dnn_init(libxsmm_target_archid);
      { const char *const env = getenv("LIBXSMM_GEMM_PREFETCH");
#if (defined(_WIN32) || defined(__CYGWIN__))
        libxsmm_gemm_auto_prefetch_default = INTERNAL_PREFETCH;
#else
        libxsmm_gemm_auto_prefetch_default = (0 == internal_statistic_ntry(0/*DP*/) && 0 == internal_statistic_ntry(1/*SP*/))
          /* avoid special prefetch if static code is present, since such code uses INTERNAL_PREFETCH */
          ? (((LIBXSMM_X86_AVX512 >= libxsmm_target_archid || LIBXSMM_X86_AVX512_CORE <= libxsmm_target_archid))
            ? LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C : LIBXSMM_GEMM_PREFETCH_BL2_VIA_C)
          : INTERNAL_PREFETCH;
#endif
        libxsmm_gemm_auto_prefetch = INTERNAL_PREFETCH;
        if (NULL != env && '\0' != *env) { /* user input beyond auto-prefetch is always considered */
          const int uid = atoi(env);
          if (0 <= uid) {
            libxsmm_gemm_auto_prefetch_default = libxsmm_gemm_uid2prefetch(uid);
            libxsmm_gemm_auto_prefetch = libxsmm_gemm_auto_prefetch_default;
            internal_gemm_auto_prefetch_locked = 1;
          }
        }
      }
      for (i = 0; i < (LIBXSMM_CAPACITY_REGISTRY); ++i) ((libxsmm_code_pointer*)new_registry)[i].ptr = NULL;
      LIBXSMM_ASSERT(NULL == internal_registry && NULL == internal_registry_keys);
#if defined(LIBXSMM_NTHREADS_USE) && defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
      LIBXSMM_ASSERT(NULL == internal_cache_buffer);
      internal_cache_buffer = (internal_cache_type*)new_cache;
#endif
      internal_registry_keys = (internal_regkey_type*)new_keys; /* prior to registering static kernels */
#if defined(LIBXSMM_BUILD) && !defined(LIBXSMM_DEFAULT_CONFIG)
#     include <libxsmm_dispatch.h>
#endif
      libxsmm_gemm_init(libxsmm_target_archid);
#if defined(LIBXSMM_TRACE)
      { int filter_threadid = 0/*only main-thread*/, filter_mindepth = 0, filter_maxnsyms = 0;
        const int init_code = libxsmm_trace_init(filter_threadid, filter_mindepth, filter_maxnsyms);
        if (EXIT_SUCCESS != init_code && 0 != libxsmm_verbosity) { /* library code is expected to be mute */
          fprintf(stderr, "LIBXSMM ERROR: failed to initialize TRACE (error #%i)!\n", init_code);
        }
      }
#endif
      { /* commit the registry buffer and enable global visibility */
        void *const pv_registry = &internal_registry;
        LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, LIBXSMM_BITS)((void**)pv_registry, (void*)new_registry, LIBXSMM_ATOMIC_SEQ_CST);
      }
    }
    else {
      if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXSMM ERROR: failed to allocate internal buffers!\n");
      }
      libxsmm_xfree(new_registry, 0/*no check*/);
      libxsmm_xfree(new_keys, 0/*no check*/);
#if defined(LIBXSMM_NTHREADS_USE) && defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
      libxsmm_xfree(new_cache, 0/*no check*/);
#endif
    }
  }
#if (0 != LIBXSMM_SYNC) /* release locks */
# if (1 < INTERNAL_REGLOCK_MAXN)
  for (i = 0; i < internal_reglock_count; ++i) LIBXSMM_LOCK_RELEASE(LIBXSMM_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXSMM_UNIFY_LOCKS)
  LIBXSMM_LOCK_RELEASE(LIBXSMM_REGLOCK, internal_reglock_ptr);
# endif
  LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK, &libxsmm_lock_global);
#endif
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_CTOR void libxsmm_init(void)
{
  if (0 == LIBXSMM_ATOMIC_LOAD(&internal_registry, LIBXSMM_ATOMIC_RELAXED)) {
    static unsigned int ninit = 0, gid = 0;
    const unsigned int tid = LIBXSMM_ATOMIC_ADD_FETCH(&ninit, 1, LIBXSMM_ATOMIC_SEQ_CST);
    LIBXSMM_ASSERT(0 < tid);
    /* libxsmm_ninit (1: initialization started, 2: library initialized, higher: to invalidate code-TLS) */
    if (1 == tid) {
      libxsmm_timer_tickint s0 = libxsmm_timer_tick_rtc(); /* warm-up */
      libxsmm_timer_tickint t0 = libxsmm_timer_tick_tsc(); /* warm-up */
      s0 = libxsmm_timer_tick_rtc(); t0 = libxsmm_timer_tick_tsc(); /* start timing */
      assert(0 == LIBXSMM_ATOMIC_LOAD(&libxsmm_ninit, LIBXSMM_ATOMIC_SEQ_CST)); /* !LIBXSMM_ASSERT */
      /* coverity[check_return] */
      LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_ninit, 1, LIBXSMM_ATOMIC_SEQ_CST);
      gid = tid; /* protect initialization */
#if (0 != LIBXSMM_SYNC)
      /* coverity[check_return] */
      LIBXSMM_TLS_CREATE(&libxsmm_tlskey);
      { /* construct and initialize locks */
# if defined(LIBXSMM_REGLOCK_TRY)
        const char *const env_trylock = getenv("LIBXSMM_TRYLOCK");
# endif
        LIBXSMM_LOCK_ATTR_TYPE(LIBXSMM_LOCK) attr_global;
# if (1 < INTERNAL_REGLOCK_MAXN)
        int i;
        LIBXSMM_LOCK_ATTR_TYPE(LIBXSMM_REGLOCK) attr;
        LIBXSMM_LOCK_ATTR_INIT(LIBXSMM_REGLOCK, &attr);
# elif defined(LIBXSMM_UNIFY_LOCKS)
        internal_reglock_ptr = &libxsmm_lock_global;
# else
        static LIBXSMM_LOCK_TYPE(LIBXSMM_REGLOCK) internal_reglock;
        internal_reglock_ptr = &internal_reglock;
        LIBXSMM_LOCK_ATTR_TYPE(LIBXSMM_REGLOCK) attr;
        LIBXSMM_LOCK_ATTR_INIT(LIBXSMM_REGLOCK, &attr);
        LIBXSMM_LOCK_INIT(LIBXSMM_REGLOCK, internal_reglock_ptr, &attr);
        LIBXSMM_LOCK_ATTR_DESTROY(LIBXSMM_REGLOCK, &attr);
# endif
        LIBXSMM_LOCK_ATTR_INIT(LIBXSMM_LOCK, &attr_global);
        LIBXSMM_LOCK_INIT(LIBXSMM_LOCK, &libxsmm_lock_global, &attr_global);
        LIBXSMM_LOCK_ATTR_DESTROY(LIBXSMM_LOCK, &attr_global);
        /* control number of locks needed; LIBXSMM_TRYLOCK implies only 1 lock */
# if defined(LIBXSMM_REGLOCK_TRY)
        if (NULL == env_trylock || 0 == *env_trylock)
# endif
        { /* no LIBXSMM_TRYLOCK */
# if defined(LIBXSMM_VTUNE)
          internal_reglock_count = 1; /* avoid duplicated kernels */
# elif (1 < INTERNAL_REGLOCK_MAXN)
          const char *const env_nlocks = getenv("LIBXSMM_NLOCKS");
          const int reglock_count = (NULL == env_nlocks || 0 == *env_nlocks || 1 > atoi(env_nlocks))
            ? (INTERNAL_REGLOCK_MAXN) : LIBXSMM_MIN(atoi(env_nlocks), INTERNAL_REGLOCK_MAXN);
          internal_reglock_count = LIBXSMM_LO2POT(reglock_count);
# else
          internal_reglock_count = 0;
# endif
        }
# if defined(LIBXSMM_REGLOCK_TRY)
        else { /* LIBXSMM_TRYLOCK environment variable specified */
          internal_reglock_count = (0 != atoi(env_trylock) ? 1
#   if (1 < INTERNAL_REGLOCK_MAXN)
            : INTERNAL_REGLOCK_MAXN);
#   else
            : 0);
#   endif
        }
# endif
# if (1 < INTERNAL_REGLOCK_MAXN)
        LIBXSMM_ASSERT(1 <= internal_reglock_count);
        for (i = 0; i < internal_reglock_count; ++i) LIBXSMM_LOCK_INIT(LIBXSMM_REGLOCK, &internal_reglock[i].state, &attr);
        LIBXSMM_LOCK_ATTR_DESTROY(LIBXSMM_REGLOCK, &attr);
# endif
      }
#endif
      { /* determine whether this instance is unique or not */
#if defined(_WIN32)
        internal_singleton_handle = CreateMutex(NULL, TRUE, "GlobalLIBXSMM");
#else
        const int result = LIBXSMM_SNPRINTF(internal_singleton_fname, sizeof(internal_singleton_fname), "/tmp/.libxsmm.%u",
          /*rely on user id to avoid permission issues in case of left-over files*/(unsigned int)getuid());
        struct flock singleton_flock;
        int singleton_handle;
        singleton_flock.l_start = 0;
        singleton_flock.l_len = 0; /* entire file */
        singleton_flock.l_type = F_WRLCK; /* exclusive across PIDs */
        singleton_flock.l_whence = SEEK_SET;
        singleton_handle = ((0 < result && (int)sizeof(internal_singleton_fname) > result) ? open(
          internal_singleton_fname, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR) : -1);
        internal_singleton_handle = fcntl(singleton_handle, F_SETLK, &singleton_flock);
        if (0 > internal_singleton_handle && 0 <= singleton_handle) close(singleton_handle);
#endif  /* coverity[leaked_handle] */
      }
      { /* calibrate timer */
        int register_termination_proc;
        libxsmm_timer_tickint s1, t1;
        internal_init(); /* must be first to initialize verbosity, etc. */
        if (INTERNAL_SINGLETON(internal_singleton_handle)) { /* after internal_init */
          internal_dump(stdout, 1/*urgent*/);
        }
        s1 = libxsmm_timer_tick_rtc(); t1 = libxsmm_timer_tick_tsc(); /* mid-timing */
#if defined(LIBXSMM_PLATFORM_X86)
        libxsmm_cpuid_x86(&internal_cpuid_info);
        if (0 != internal_cpuid_info.constant_tsc && t0 < t1) {
          libxsmm_timer_scale = libxsmm_timer_duration_rtc(s0, s1) / (t1 - t0);
        }
#endif
        register_termination_proc = atexit(internal_finalize);
        s1 = libxsmm_timer_tick_rtc(); t1 = libxsmm_timer_tick_tsc(); /* final timing */
        /* set timer-scale and determine start of the "uptime" (shown at termination) */
        if (t0 < t1 && 0.0 < libxsmm_timer_scale) {
          const double scale = libxsmm_timer_duration_rtc(s0, s1) / (t1 - t0);
          const double diff = LIBXSMM_DELTA(libxsmm_timer_scale, scale) / scale;
          if (5E-4 > diff) {
            libxsmm_timer_scale = scale;
            internal_timer_start = t0;
          }
          else {
            libxsmm_timer_scale = 0;
            internal_timer_start = s0;
#if defined(_DEBUG)
            libxsmm_se = 1;
#endif
          }
        }
        else {
          internal_timer_start = s0;
          libxsmm_timer_scale = 0;
        }
        if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
          if (EXIT_SUCCESS != register_termination_proc) {
            fprintf(stderr, "LIBXSMM ERROR: failed to register termination procedure!\n");
          }
          if (0 == libxsmm_timer_scale
#if defined(LIBXSMM_PLATFORM_X86)
            && 0 == internal_cpuid_info.constant_tsc
#endif
            && (LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity))
          {
            /* ARM: TSC is currently not implemented, hence warning shows up (if verbose) */
            fprintf(stderr, "LIBXSMM WARNING: timer is maybe not cycle-accurate!\n");
          }
        }
      }
      assert(1 == LIBXSMM_ATOMIC_LOAD(&libxsmm_ninit, LIBXSMM_ATOMIC_SEQ_CST)); /* !LIBXSMM_ASSERT */
      /* coverity[check_return] */
      LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_ninit, 1, LIBXSMM_ATOMIC_SEQ_CST);
    }
    else /*if (gid != tid)*/ { /* avoid recursion */
      LIBXSMM_ASSERT(gid != tid);
      LIBXSMM_UNUSED(gid);
      while (2 > LIBXSMM_ATOMIC_LOAD(&libxsmm_ninit, LIBXSMM_ATOMIC_RELAXED)) LIBXSMM_SYNC_YIELD;
      internal_init();
    }
#if defined(LIBXSMM_PERF)
    libxsmm_perf_init();
#endif
  }
  LIBXSMM_ASSERT(1 < libxsmm_ninit);
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_NO_TRACE void libxsmm_finalize(void);
LIBXSMM_API LIBXSMM_ATTRIBUTE_DTOR void libxsmm_finalize(void)
{
  void *const regaddr = &internal_registry;
  uintptr_t regptr = LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_LOAD, LIBXSMM_BITS)((uintptr_t*)regaddr, LIBXSMM_ATOMIC_RELAXED);
  libxsmm_code_pointer* registry = (libxsmm_code_pointer*)regptr;
  if (NULL != registry) {
    int i;
#if (0 != LIBXSMM_SYNC)
    LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK, &libxsmm_lock_global);
# if (1 < INTERNAL_REGLOCK_MAXN)
    { /* acquire locks and thereby shortcut lazy initialization later on */
      int ntry = 0, n;
      do {
        for (i = 0, n = 0; i < internal_reglock_count; ++i) {
          if (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_REGLOCK) == LIBXSMM_LOCK_TRYLOCK(LIBXSMM_REGLOCK, &internal_reglock[i].state)) ++n;
        }
        ntry += (0 == n ? 1 : 0);
      } while (n < internal_reglock_count && ntry < LIBXSMM_CLEANUP_NTRY);
    }
# elif !defined(LIBXSMM_UNIFY_LOCKS)
    LIBXSMM_LOCK_ACQUIRE(LIBXSMM_REGLOCK, internal_reglock_ptr);
# endif
#endif
    regptr = LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_LOAD, LIBXSMM_BITS)((uintptr_t*)regaddr, LIBXSMM_ATOMIC_RELAXED);
    registry = (libxsmm_code_pointer*)regptr;
    if (NULL != registry) {
      internal_regkey_type *const registry_keys = internal_registry_keys;
#if defined(LIBXSMM_NTHREADS_USE) && defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
      internal_cache_type *const cache_buffer = internal_cache_buffer;
#endif
      unsigned int rest = 0, errors = 0;
#if defined(LIBXSMM_TRACE)
      i = libxsmm_trace_finalize();
      if (EXIT_SUCCESS != i && 0 != libxsmm_verbosity) { /* library code is expected to be mute */
        fprintf(stderr, "LIBXSMM ERROR: failed to finalize trace (error #%i)!\n", i);
      }
#endif
#if defined(LIBXSMM_PERF)
      libxsmm_perf_finalize();
#endif
      libxsmm_xcopy_finalize();
      libxsmm_gemm_finalize();
      libxsmm_dnn_finalize();
      /* coverity[check_return] */
      LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_ninit, 1, LIBXSMM_ATOMIC_RELAXED); /* invalidate code cache (TLS) */
#if defined(LIBXSMM_NTHREADS_USE) && defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
      internal_cache_buffer = NULL;
#endif
      internal_registry_keys = NULL; /* make registry keys unavailable */
      LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE_ZERO, LIBXSMM_BITS)((uintptr_t*)regaddr, LIBXSMM_ATOMIC_SEQ_CST);
      internal_registry_nbytes = 0; internal_registry_nleaks = 0;
      for (i = 0; i < (LIBXSMM_CAPACITY_REGISTRY); ++i) {
        /*const*/ libxsmm_code_pointer code = registry[i];
        if (NULL != code.ptr_const) {
          /* check if the registered entity is a GEMM kernel */
          switch (LIBXSMM_DESCRIPTOR_KIND(registry_keys[i].entry.kind)) {
            case LIBXSMM_KERNEL_KIND_MATMUL: {
              const libxsmm_gemm_descriptor *const desc = &registry_keys[i].entry.gemm.desc;
              if (1 < desc->m && 1 < desc->n) {
                const unsigned int njit = (0 == (LIBXSMM_CODE_STATIC & code.uval) ? 1 : 0);
                const unsigned int nsta = (0 != (LIBXSMM_CODE_STATIC & code.uval) ? 1 : 0);
                /* count whether kernel is static or JIT-code */
                internal_update_mmstatistic(desc, 0, 0, njit, nsta);
              }
              else {
                ++internal_statistic_num_gemv;
              }
              ++rest;
            } break;
            case LIBXSMM_KERNEL_KIND_MELTW: {
              ++internal_statistic_num_meltw;
            } break;
            case LIBXSMM_KERNEL_KIND_USER: {
              ++internal_statistic_num_user;
            } break;
            default: if (LIBXSMM_KERNEL_UNREGISTERED <= LIBXSMM_DESCRIPTOR_KIND(registry_keys[i].entry.kind)) {
              ++errors;
            }
            else {
              ++rest;
            }
          }
          if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
            if (0 != errors) {
              fprintf(stderr, "LIBXSMM ERROR: code registry is corrupted!\n");
            }
            if (LIBXSMM_CAPACITY_REGISTRY == (rest + errors + internal_statistic_num_gemv +
              internal_statistic_num_user + internal_statistic_num_meltw))
            {
              fprintf(stderr, "LIBXSMM WARNING: code registry was exhausted!\n");
            }
          }
          if (0 == (LIBXSMM_CODE_STATIC & code.uval)) { /* check for allocated/generated JIT-code */
# if defined(__APPLE__) && defined(__arm64__)
# else
            void* buffer = NULL;
            size_t size = 0;
# endif
#if defined(LIBXSMM_HASH_COLLISION)
            code.uval &= ~LIBXSMM_HASH_COLLISION; /* clear collision flag */
#endif
# if defined(__APPLE__) && defined(__arm64__)
            ++internal_registry_nleaks;
#else
            if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(code.ptr_const, &size, NULL/*flags*/, &buffer)) {
              if (LIBXSMM_KERNEL_KIND_USER == LIBXSMM_DESCRIPTOR_KIND(registry_keys[i].entry.kind)
                /* dump user-data just like JIT'ted code */
                && 0 > libxsmm_verbosity)
              {
                char name[16];
                int nchar;
#if defined(LIBXSMM_REGUSER_HASH)
                const size_t descsize = LIBXSMM_DESCRIPTOR_ISBIG(registry_keys[i].entry.kind)
                  ? LIBXSMM_DESCRIPTOR_MAXSIZE : LIBXSMM_DESCRIPTOR_SIGSIZE;
                const unsigned int id = libxsmm_crc32(LIBXSMM_HASH_SEED, registry_keys[i].entry.user.desc,
                  descsize - sizeof(libxsmm_descriptor_kind));
                LIBXSMM_ASSERT(descsize > sizeof(libxsmm_descriptor_kind));
#else
                const unsigned int id = internal_statistic_num_user;
#endif
                nchar = LIBXSMM_SNPRINTF(name, sizeof(name), "%010u.user", id);
                if (0 < nchar && (int)sizeof(name) > nchar) {
                  LIBXSMM_EXPECT(EXIT_SUCCESS, libxsmm_dump("LIBXSMM-USER-DUMP", name, code.ptr_const, size, 0/*unique*/));
                }
              }
#if !defined(NDEBUG)
              registry[i].ptr = NULL;
#endif
              libxsmm_xfree(code.ptr_const, 0/*no check*/);
              /* round-up size (it is fine to assume 4 KB pages since it is likely more accurate than not rounding up) */
              internal_registry_nbytes += LIBXSMM_UP2(size + (((char*)code.ptr_const) - (char*)buffer), LIBXSMM_PAGE_MINSIZE);
            }
            else ++internal_registry_nleaks;
#endif
          }
        }
      }
      /* release buffers (registry, keys, cache) */
#if defined(LIBXSMM_NTHREADS_USE) && defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
      libxsmm_xfree(cache_buffer, 0/*no check*/);
#endif
      libxsmm_xfree(registry_keys, 0/*no check*/);
      libxsmm_xfree(registry, 0/*no check*/);
    }
#if (0 != LIBXSMM_SYNC) /* LIBXSMM_LOCK_RELEASE, but no LIBXSMM_LOCK_DESTROY */
# if (1 < INTERNAL_REGLOCK_MAXN)
    for (i = 0; i < internal_reglock_count; ++i) LIBXSMM_LOCK_RELEASE(LIBXSMM_REGLOCK, &internal_reglock[i].state);
# elif !defined(LIBXSMM_UNIFY_LOCKS)
    LIBXSMM_LOCK_RELEASE(LIBXSMM_REGLOCK, internal_reglock_ptr);
# endif
    LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK, &libxsmm_lock_global);
    /* coverity[check_return] */
    LIBXSMM_TLS_DESTROY(libxsmm_tlskey);
#endif
  }
}


LIBXSMM_API void libxsmm_sink(LIBXSMM_VARIADIC)
{
  /* does nothing else but sinking given arguments */
}


LIBXSMM_API int libxsmm_get_target_archid(void)
{
  LIBXSMM_INIT
#if !defined(__MIC__)
  return libxsmm_target_archid;
#else /* no JIT support */
  return LIBXSMM_MIN(libxsmm_target_archid, LIBXSMM_X86_GENERIC);
#endif
}


LIBXSMM_API void libxsmm_set_target_archid(int id)
{
  int target_archid = LIBXSMM_TARGET_ARCH_UNKNOWN;
  switch (id) {
    case LIBXSMM_X86_AVX512_SPR:
    case LIBXSMM_X86_AVX512_CPX:
    case LIBXSMM_X86_AVX512_CLX:
    case LIBXSMM_X86_AVX512_CORE:
    case LIBXSMM_X86_AVX512_KNM:
    case LIBXSMM_X86_AVX512_MIC:
    case LIBXSMM_X86_AVX512:
    case LIBXSMM_X86_AVX512_VL256:
    case LIBXSMM_X86_AVX512_VL256_CLX:
    case LIBXSMM_X86_AVX512_VL256_CPX:
    case LIBXSMM_X86_AVX2:
    case LIBXSMM_X86_AVX:
    case LIBXSMM_X86_SSE42:
    case LIBXSMM_X86_SSE3:
    case LIBXSMM_AARCH64_V81:
    case LIBXSMM_AARCH64_V82:
    case LIBXSMM_AARCH64_A64FX: {
      target_archid = id;
    } break;
    case LIBXSMM_TARGET_ARCH_GENERIC:
#if defined(LIBXSMM_PLATFORM_X86)
      target_archid = LIBXSMM_X86_GENERIC;
      break;
#elif defined(LIBXSMM_PLATFORM_AARCH64)
      target_archid = LIBXSMM_AARCH64_V81;
      break;
#endif
    default: target_archid = libxsmm_cpuid();
  }
  LIBXSMM_ATOMIC_STORE(&libxsmm_target_archid, target_archid, LIBXSMM_ATOMIC_RELAXED);
  if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    const int cpuid = libxsmm_cpuid();
    if (cpuid < target_archid) {
      const char *const target_arch = libxsmm_cpuid_name(target_archid);
      fprintf(stderr, "LIBXSMM WARNING: \"%s\" code may fail to run on \"%s\"!\n",
        target_arch, libxsmm_cpuid_name(cpuid));
    }
  }
}


LIBXSMM_API const char* libxsmm_get_target_arch(void)
{
  LIBXSMM_INIT
  return libxsmm_cpuid_name(libxsmm_target_archid);
}


/* function serves as a helper for implementing the Fortran interface */
LIBXSMM_API const char* libxsmmf_get_target_arch(int* length);
LIBXSMM_API const char* libxsmmf_get_target_arch(int* length)
{
  const char *const arch = libxsmm_get_target_arch();
  /* valid here since function is not in the public interface */
  LIBXSMM_ASSERT(NULL != arch && 0 != length);
  *length = (int)strlen(arch);
  return arch;
}


LIBXSMM_API void libxsmm_set_target_arch(const char* arch)
{
  const int cpuid = libxsmm_cpuid();
  int target_archid;
  if (NULL != arch && '\0' != *arch) {
#if defined(LIBXSMM_PLATFORM_X86)
    const int jit = atoi(arch);
#endif
    if (0 == strcmp("0", arch)) {
#if defined(LIBXSMM_PLATFORM_X86)
      target_archid = LIBXSMM_X86_GENERIC;
#elif defined(LIBXSMM_PLATFORM_AARCH64)
      target_archid = LIBXSMM_AARCH64_V81;
#else
      target_archid = LIBXSMM_TARGET_ARCH_GENERIC;
#endif
    }
#if defined(LIBXSMM_PLATFORM_X86)
    else if (0 < jit) {
      target_archid = LIBXSMM_X86_GENERIC + jit;
    }
    else if (arch == libxsmm_stristr(arch, "avx512_vl256_cpx")) {
      target_archid = LIBXSMM_X86_AVX512_VL256_CPX;
    }
    else if (arch == libxsmm_stristr(arch, "avx512_vl256_clx")) {
      target_archid = LIBXSMM_X86_AVX512_VL256_CLX;
    }
    else if (arch == libxsmm_stristr(arch, "avx512_vl256")) {
      target_archid = LIBXSMM_X86_AVX512_VL256;
    }
    else if (arch == libxsmm_stristr(arch, "spr") || arch == libxsmm_stristr(arch, "amx")) {
      target_archid = LIBXSMM_X86_AVX512_SPR;
    }
    else if (arch == libxsmm_stristr(arch, "cpx")) {
      target_archid = LIBXSMM_X86_AVX512_CPX;
    }
    else if (arch == libxsmm_stristr(arch, "clx")) {
      target_archid = LIBXSMM_X86_AVX512_CLX;
    }
    else if (arch == libxsmm_stristr(arch, "skx") || arch == libxsmm_stristr(arch, "skl")
          /* "avx3"/"avx512" previously enabled LIBXSMM_X86_AVX512 */
          || arch == libxsmm_stristr(arch, "avx3") || arch == libxsmm_stristr(arch, "avx512"))
    {
      target_archid = LIBXSMM_X86_AVX512_CORE;
    }
    else if (arch == libxsmm_stristr(arch, "knm")) {
      target_archid = LIBXSMM_X86_AVX512_KNM;
    }
    else if (arch == libxsmm_stristr(arch, "knl") || arch == libxsmm_stristr(arch, "mic")) {
      target_archid = LIBXSMM_X86_AVX512_MIC;
    }
    else if (arch == libxsmm_stristr(arch, "hsw") || arch == libxsmm_stristr(arch, "avx2")) {
      target_archid = LIBXSMM_X86_AVX2;
    }
    else if (arch == libxsmm_stristr(arch, "snb") || arch == libxsmm_stristr(arch, "avx")) {
      target_archid = LIBXSMM_X86_AVX;
    }
    else if (arch == libxsmm_stristr(arch, "wsm") || arch == libxsmm_stristr(arch, "nhm")
       || arch == libxsmm_stristr(arch, "sse4_2") || arch == libxsmm_stristr(arch, "sse4.2")
       || arch == libxsmm_stristr(arch, "sse42")  || arch == libxsmm_stristr(arch, "sse4"))
    {
      target_archid = LIBXSMM_X86_SSE42;
    }
    else if (arch == libxsmm_stristr(arch, "sse3")) {
      target_archid = LIBXSMM_X86_SSE3;
    }
    else if (arch == libxsmm_stristr(arch, "x86") || arch == libxsmm_stristr(arch, "x86_64")
          || arch == libxsmm_stristr(arch, "x64") || arch == libxsmm_stristr(arch, "sse2"))
    {
      target_archid = LIBXSMM_X86_GENERIC;
    }
#elif defined(LIBXSMM_PLATFORM_AARCH64)
    else if (arch == libxsmm_stristr(arch, "arm") || arch == libxsmm_stristr(arch, "arm64")
          || arch == libxsmm_stristr(arch, "arm_v81")
          || arch == libxsmm_stristr(arch, "aarch64"))
    {
      target_archid = LIBXSMM_AARCH64_V81;
    }
    else if (arch == libxsmm_stristr(arch, "arm_v82")) {
      target_archid = LIBXSMM_AARCH64_V82;
    }
    else if (arch == libxsmm_stristr(arch, "a64fx"))
    {
      target_archid = LIBXSMM_AARCH64_A64FX;
    }
#endif
    else if (arch == libxsmm_stristr(arch, "generic")) {
#if defined(LIBXSMM_PLATFORM_X86)
      target_archid = LIBXSMM_X86_GENERIC;
#elif defined(LIBXSMM_PLATFORM_AARCH64)
      target_archid = LIBXSMM_AARCH64_V81;
#else
      target_archid = LIBXSMM_TARGET_ARCH_GENERIC;
#endif
    }
    else if (arch == libxsmm_stristr(arch, "none")) {
      target_archid = LIBXSMM_TARGET_ARCH_GENERIC;
    }
    else {
      target_archid = cpuid;
    }
  }
  else {
    target_archid = cpuid;
  }
  if (cpuid < target_archid) { /* warn about code path if beyond CPUID */
    static int error_once = 0;
    if ( 0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      const char *const target_arch = libxsmm_cpuid_name(target_archid);
      fprintf(stderr, "LIBXSMM WARNING: \"%s\" code will fail to run on \"%s\"!\n",
        target_arch, libxsmm_cpuid_name(cpuid));
    }
#if 0 /* limit code path to confirmed features */
    target_archid = cpuid;
#endif
  }
  LIBXSMM_ATOMIC_STORE(&libxsmm_target_archid, target_archid, LIBXSMM_ATOMIC_RELAXED);
}


LIBXSMM_API int libxsmm_get_verbosity(void)
{
  LIBXSMM_INIT
  return libxsmm_verbosity;
}


LIBXSMM_API void libxsmm_set_verbosity(int level)
{
  LIBXSMM_INIT
  LIBXSMM_ATOMIC_STORE(&libxsmm_verbosity, level, LIBXSMM_ATOMIC_RELAXED);
}


LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_auto_prefetch(void)
{
  return (libxsmm_gemm_prefetch_type)libxsmm_gemm_auto_prefetch;
}


LIBXSMM_API void libxsmm_set_gemm_auto_prefetch(libxsmm_gemm_prefetch_type strategy)
{
  if (0 == internal_gemm_auto_prefetch_locked) { /* LIBXSMM_GEMM_PREFETCH environment takes precedence */
    LIBXSMM_ATOMIC_STORE(&libxsmm_gemm_auto_prefetch_default, strategy, LIBXSMM_ATOMIC_RELAXED);
    LIBXSMM_ATOMIC_STORE(&libxsmm_gemm_auto_prefetch, strategy, LIBXSMM_ATOMIC_RELAXED);
  }
}


LIBXSMM_API unsigned char libxsmm_typesize(libxsmm_datatype datatype)
{
  const unsigned char result = (unsigned char)LIBXSMM_TYPESIZE(datatype);
  if (0 != result) {
    return result;
  }
  else {
    static int error_once = 0;
    LIBXSMM_ASSERT_MSG(0, "unsupported data type");
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM ERROR: unsupported data type!\n");
    }
    return 1; /* avoid to return 0 to avoid div-by-zero in static analysis of depending code */
  }
}


LIBXSMM_API int libxsmm_dvalue(libxsmm_datatype datatype, const void* value, double* dvalue)
{
  int result = EXIT_SUCCESS;
  if (NULL != value && NULL != dvalue) {
    switch (datatype) {
      case LIBXSMM_DATATYPE_F64: *dvalue =         (*(const double   *)value); break;
      case LIBXSMM_DATATYPE_F32: *dvalue = (double)(*(const float    *)value); break;
      case LIBXSMM_DATATYPE_I64: *dvalue = (double)(*(const long long*)value); break;
      case LIBXSMM_DATATYPE_I32: *dvalue = (double)(*(const int      *)value); break;
      case LIBXSMM_DATATYPE_I16: *dvalue = (double)(*(const short    *)value); break;
      case LIBXSMM_DATATYPE_I8:  *dvalue = (double)(*(const char     *)value); break;
      default: result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_INTERN const char* libxsmm_typename(libxsmm_datatype datatype)
{
  switch (datatype) {
    case LIBXSMM_DATATYPE_F64:  return "f64";
    case LIBXSMM_DATATYPE_F32:  return "f32";
    case LIBXSMM_DATATYPE_BF16: return "bf16";
    case LIBXSMM_DATATYPE_F16:  return "f16";
    case LIBXSMM_DATATYPE_I64:  return "i64";
    case LIBXSMM_DATATYPE_I32:  return "i32";
    case LIBXSMM_DATATYPE_I16:  return "i16";
    case LIBXSMM_DATATYPE_I8:   return "i8";
    default: {
      if (LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP(datatype) &&
          LIBXSMM_GEMM_PRECISION_I32 == LIBXSMM_GETENUM_OUT(datatype))
      {
        return "i16i32";
      }
      else if (LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP(datatype) &&
               LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT(datatype))
      {
        return "i16f32";
      }
      else if (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP(datatype) &&
               LIBXSMM_GEMM_PRECISION_I32 == LIBXSMM_GETENUM_OUT(datatype))
      {
        return "i8i32";
      }
      else if (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP(datatype) &&
               LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT(datatype))
      {
        return "bf16f32";
      }
      else {
        return "void";
      }
    }
  }
}


LIBXSMM_API_INLINE void internal_get_typesize_string(char buffer[4], int buffer_size, size_t typesize)
{
  LIBXSMM_ASSERT(256 > typesize && 4 <= buffer_size);
  if (10 > typesize) {
    buffer[0] = (char)('0' + typesize);
    buffer[1] = 0;
  }
  else {
    LIBXSMM_SNPRINTF(buffer, buffer_size, "%i", (int)typesize);
  }
}


LIBXSMM_API_INTERN int libxsmm_dump(const char* title, const char* name, const void* data, size_t size, int unique)
{
  int result;
  if (NULL != name && '\0' != *name && NULL != data && 0 != size) {
    FILE* data_file = fopen(name, "rb");
    int diff = 0, result_close;
    if (NULL == data_file) { /* file does not exist */
      data_file = fopen(name, "wb");
      if (NULL != data_file) { /* dump data into a file */
        result = ((size == fwrite(data, 1, size, data_file)) ? EXIT_SUCCESS : EXIT_FAILURE);
        result_close = fclose(data_file);
        if (EXIT_SUCCESS == result) result = result_close;
      }
      else result = EXIT_FAILURE;
    }
    else if (0 != unique) { /* check existing file */
      const char* check_a = (const char*)data;
      char check_b[4096];
      size_t rest = size;
      do {
        const size_t n = fread(check_b, 1, LIBXSMM_MIN(sizeof(check_b), rest), data_file);
        diff += memcmp(check_a, check_b, LIBXSMM_MIN(sizeof(check_b), n));
        check_a += n;
        rest -= n;
      } while (0 < rest && 0 == diff);
      result = fclose(data_file);
    }
    else {
      result = fclose(data_file);
    }
    if (EXIT_SUCCESS == result && NULL != title && '\0' != *title) {
      fprintf(stderr, "%s(ptr:file) %p : %s\n", title, data, name);
    }
    if (0 != diff) { /* override existing dump and warn about erroneous condition */
      fprintf(stderr, "LIBXSMM ERROR: %s is not a unique filename!\n", name);
      data_file = fopen(name, "wb");
      if (NULL != data_file) { /* dump data into a file */
        if (size != fwrite(data, 1, size, data_file)) result = EXIT_FAILURE;
        result_close = fclose(data_file);
        if (EXIT_SUCCESS == result) result = result_close;
      }
      if (EXIT_SUCCESS == result) result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_INTERN int libxsmm_build(const libxsmm_build_request* request, unsigned int regindex, libxsmm_code_pointer* code)
{
  int result = EXIT_SUCCESS;
#if !defined(__MIC__)
  const char * /*const*/ target_arch = libxsmm_cpuid_name(libxsmm_target_archid);
  /* large enough temporary buffer for generated code */
  char jit_buffer[LIBXSMM_CODE_MAXSIZE], jit_name[256] = { 0 };
  libxsmm_generated_code generated_code;
  libxsmm_kernel_xinfo extra;

  LIBXSMM_MEMZERO127(&generated_code);
  generated_code.generated_code = jit_buffer;
  generated_code.buffer_size = sizeof(jit_buffer);
  /* setup code generation */
  generated_code.arch = libxsmm_target_archid;
  generated_code.code_type = 2;

# if !defined(NDEBUG) /* should not be needed (all members will be initialized below) */
  LIBXSMM_MEMZERO127(&extra);
# endif
  extra.registered = regindex;
  extra.nflops = 0;

  LIBXSMM_ASSERT(NULL != generated_code.generated_code || 0 == generated_code.buffer_size);
  LIBXSMM_ASSERT(NULL != request && 0 != libxsmm_target_archid);
  LIBXSMM_ASSERT(NULL != code && NULL == code->ptr_const);
  LIBXSMM_ASSERT(0 == LIBXSMM_DESCRIPTOR_ISBIG(request->kind));

  switch (request->kind) { /* generate kernel */
    case LIBXSMM_BUILD_KIND_GEMM: { /* small MxM kernel */
      LIBXSMM_ASSERT(NULL != request->descriptor.gemm);
# if 0 /* dummy kernel for an empty shape is desired */
      if (0 < request->descriptor.gemm->m   && 0 < request->descriptor.gemm->n   && 0 < request->descriptor.gemm->k &&
          0 < request->descriptor.gemm->lda && 0 < request->descriptor.gemm->ldb && 0 < request->descriptor.gemm->ldc)
# endif
      {
        const unsigned int m = request->descriptor.gemm->m, n = request->descriptor.gemm->n, k = request->descriptor.gemm->k;
        extra.nflops = 2 * m * n * k;
# if !defined(LIBXSMM_DENY_RETARGET) /* disable: ECFLAGS=-DLIBXSMM_DENY_RETARGET */
        if ((LIBXSMM_X86_AVX2 < libxsmm_target_archid) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT) &&
           (LIBXSMM_GEMM_PRECISION_F64 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.gemm->datatype) ||
            LIBXSMM_GEMM_PRECISION_F32 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.gemm->datatype)) &&
           (16 >= (m * k) || 16 >= (k * n) || 16 >= (m * n)))
        {
          /* TODO: shall we update variable "target_arch" (name)? */
          generated_code.arch = LIBXSMM_X86_AVX2;
        }
# endif
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_gemm_kernel, &generated_code, request->descriptor.gemm);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid((libxsmm_gemm_prefetch_type)request->descriptor.gemm->prefetch);
          const char *const tname = libxsmm_typename((libxsmm_datatype)request->descriptor.gemm->datatype);
          const char *const meltw_tname = libxsmm_typename((libxsmm_datatype)request->descriptor.gemm->meltw_datatype_aux);
          int typesigns = 0, br = 0;
          char tc_option[16] = { 0 };
          int decompress_A = 0;
          int sparsity_factor_A = 1;
          if ((request->descriptor.gemm->meltw_operation == LIBXSMM_MELTW_OPERATION_DECOMPRESS_A) ||
              (request->descriptor.gemm->meltw_operation == LIBXSMM_MELTW_OPERATION_COLBIAS_ACT_DECOMPRESS_A))
          {
            decompress_A = 1;
            sparsity_factor_A = (int)request->descriptor.gemm->meltw_param;
          }

          /* query batch reduce variant */
          if ( (LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS & request->descriptor.gemm->flags) > 1 ) {
            br = 1;
          } else if ( (LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET & request->descriptor.gemm->flags) > 1 ) {
            br = 2;
          } else if ( (LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE & request->descriptor.gemm->flags) > 1 ) {
            br = 3;
          } else {
            br = 0;
          }
          /* query A/B sign combinations */
          if ( (LIBXSMM_GEMM_FLAG_A_UNSIGNED & request->descriptor.gemm->flags) > 1 ) {
            typesigns = 1;
          } else if ( (LIBXSMM_GEMM_FLAG_B_UNSIGNED & request->descriptor.gemm->flags) > 1 ) {
            typesigns = 2;
          } else if ( (LIBXSMM_GEMM_FLAG_AB_UNSIGNED & request->descriptor.gemm->flags) > 1 ) {
            typesigns = 3;
          } else {
            typesigns = 0;
          }
          /* query tileconfig options */
          if (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & request->descriptor.gemm->flags) != 0) &&
              ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & request->descriptor.gemm->flags) == 0) ) {
            LIBXSMM_SNPRINTF(tc_option, sizeof(tc_option), "conf");
          } else if (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & request->descriptor.gemm->flags) == 0) &&
                     ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & request->descriptor.gemm->flags) != 0) ) {
            LIBXSMM_SNPRINTF(tc_option, sizeof(tc_option), "rele");
          } else if (((LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG & request->descriptor.gemm->flags) != 0) &&
                     ((LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG & request->descriptor.gemm->flags) != 0)) {
            LIBXSMM_SNPRINTF(tc_option, sizeof(tc_option), "none");
          } else {
            LIBXSMM_SNPRINTF(tc_option, sizeof(tc_option), "abid");
          }

          if ( request->descriptor.gemm->meltw_operation != 0 ) {
            /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
            LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i_br%i_uh%u_si%i_tc-%s_avnni%i_bvnni%i_cvnni%i_meop%u-%s_mefl%u_meld%u-%u-%u_decompress_A%i_spfactor%i.mxm", target_arch, tname,
              0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.gemm->flags) ? 'n' : 't',
              0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.gemm->flags) ? 'n' : 't', m, n, k,
              request->descriptor.gemm->lda, request->descriptor.gemm->ldb, request->descriptor.gemm->ldc,
              1, 0 != (LIBXSMM_GEMM_FLAG_BETA_0  & request->descriptor.gemm->flags) ? 0 : 1, uid,
              br, (unsigned int)request->descriptor.gemm->c3, typesigns, tc_option,
              0 != (LIBXSMM_GEMM_FLAG_VNNI_A  & request->descriptor.gemm->flags) ? 1 : 0,
              0 != (LIBXSMM_GEMM_FLAG_VNNI_B  & request->descriptor.gemm->flags) ? 1 : 0,
              0 != (LIBXSMM_GEMM_FLAG_VNNI_C  & request->descriptor.gemm->flags) ? 1 : 0,
              (unsigned int)request->descriptor.gemm->meltw_operation, meltw_tname, (unsigned int)request->descriptor.gemm->meltw_flags,
              request->descriptor.gemm->meltw_ldx, request->descriptor.gemm->meltw_ldy, request->descriptor.gemm->meltw_ldz, decompress_A, sparsity_factor_A );
          } else {
            /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
            LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i_br%i_uh%u_si%i_tc-%s_avnni%i_bvnni%i_cvnni%i_decompress_A%i_spfactor%i.mxm", target_arch, tname,
              0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.gemm->flags) ? 'n' : 't',
              0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.gemm->flags) ? 'n' : 't', m, n, k,
              request->descriptor.gemm->lda, request->descriptor.gemm->ldb, request->descriptor.gemm->ldc,
              1, 0 != (LIBXSMM_GEMM_FLAG_BETA_0  & request->descriptor.gemm->flags) ? 0 : 1, uid,
              br, (unsigned int)request->descriptor.gemm->c3, typesigns, tc_option,
              0 != (LIBXSMM_GEMM_FLAG_VNNI_A  & request->descriptor.gemm->flags) ? 1 : 0,
              0 != (LIBXSMM_GEMM_FLAG_VNNI_B  & request->descriptor.gemm->flags) ? 1 : 0,
              0 != (LIBXSMM_GEMM_FLAG_VNNI_C  & request->descriptor.gemm->flags) ? 1 : 0, decompress_A, sparsity_factor_A );
          }
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_PSPGEMM_CSR: { /* packed sparse gemm kernel, CSR format */
      LIBXSMM_ASSERT(NULL != request->descriptor.pspgemm_csr && 0 != request->descriptor.pspgemm_csr->gemm);
      LIBXSMM_ASSERT(NULL != request->descriptor.pspgemm_csr->row_ptr && 0 != request->descriptor.pspgemm_csr->column_idx && 0 != request->descriptor.pspgemm_csr->values);
      /* only floating point */
      if (LIBXSMM_GEMM_PRECISION_F64 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.pspgemm_csr->gemm->datatype) ||
          LIBXSMM_GEMM_PRECISION_F32 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.pspgemm_csr->gemm->datatype))
      {
        const unsigned int nnz = (request->descriptor.pspgemm_csr->gemm->lda == 0) ?
            request->descriptor.pspgemm_csr->row_ptr[request->descriptor.pspgemm_csr->gemm->m] : request->descriptor.pspgemm_csr->row_ptr[request->descriptor.pspgemm_csr->gemm->k];
        const unsigned int gemm_factor = (request->descriptor.pspgemm_csr->gemm->lda == 0) ? request->descriptor.pspgemm_csr->gemm->n : request->descriptor.pspgemm_csr->gemm->m;
        extra.nflops = 2 * nnz * gemm_factor * request->descriptor.pspgemm_csr->packed_width;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_packed_spgemm_csr_kernel, &generated_code, request->descriptor.pspgemm_csr->gemm,
          request->descriptor.pspgemm_csr->row_ptr, request->descriptor.pspgemm_csr->column_idx, request->descriptor.pspgemm_csr->values, request->descriptor.pspgemm_csr->packed_width);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid((libxsmm_gemm_prefetch_type)request->descriptor.pspgemm_csr->gemm->prefetch);
          const char *const tname = libxsmm_typename((libxsmm_datatype)request->descriptor.pspgemm_csr->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_w%u_a%i_b%i_p%i_nnz%u.pspgemm_csr", target_arch, tname,
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.pspgemm_csr->gemm->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.pspgemm_csr->gemm->flags) ? 'n' : 't',
            request->descriptor.pspgemm_csr->gemm->m,   request->descriptor.pspgemm_csr->gemm->n,   request->descriptor.pspgemm_csr->gemm->k,
            request->descriptor.pspgemm_csr->gemm->lda, request->descriptor.pspgemm_csr->gemm->ldb, request->descriptor.pspgemm_csr->gemm->ldc,
            request->descriptor.pspgemm_csr->packed_width,
            1, 0 != (LIBXSMM_GEMM_FLAG_BETA_0  & request->descriptor.pspgemm_csr->gemm->flags) ? 0 : 1,
            uid, nnz);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_PSPGEMM_CSC: { /* packed sparse gemm kernel, CSC format */
      LIBXSMM_ASSERT(NULL != request->descriptor.pspgemm_csc && 0 != request->descriptor.pspgemm_csc->gemm);
      LIBXSMM_ASSERT(NULL != request->descriptor.pspgemm_csc->row_idx && 0 != request->descriptor.pspgemm_csc->column_ptr && 0 != request->descriptor.pspgemm_csc->values);
      /* only floating point */
      if (LIBXSMM_GEMM_PRECISION_F64 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.pspgemm_csc->gemm->datatype) ||
          LIBXSMM_GEMM_PRECISION_F32 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.pspgemm_csc->gemm->datatype))
      {
        const unsigned int nnz = (request->descriptor.pspgemm_csc->gemm->lda == 0) ?
            request->descriptor.pspgemm_csc->column_ptr[request->descriptor.pspgemm_csc->gemm->k] : request->descriptor.pspgemm_csc->column_ptr[request->descriptor.pspgemm_csc->gemm->n];
        const unsigned int gemm_factor = (request->descriptor.pspgemm_csc->gemm->lda == 0) ? request->descriptor.pspgemm_csc->gemm->n : request->descriptor.pspgemm_csc->gemm->m;
        extra.nflops = 2 * nnz * gemm_factor * request->descriptor.pspgemm_csc->packed_width;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_packed_spgemm_csc_kernel, &generated_code, request->descriptor.pspgemm_csc->gemm,
          request->descriptor.pspgemm_csc->row_idx, request->descriptor.pspgemm_csc->column_ptr, request->descriptor.pspgemm_csc->values, request->descriptor.pspgemm_csc->packed_width);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid((libxsmm_gemm_prefetch_type)request->descriptor.pspgemm_csc->gemm->prefetch);
          const char *const tname = libxsmm_typename((libxsmm_datatype)request->descriptor.pspgemm_csc->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_w%u_a%i_b%i_p%i_nnz%u.pspgemm_csc", target_arch, tname,
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.pspgemm_csc->gemm->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.pspgemm_csc->gemm->flags) ? 'n' : 't',
            request->descriptor.pspgemm_csc->gemm->m,   request->descriptor.pspgemm_csc->gemm->n,   request->descriptor.pspgemm_csc->gemm->k,
            request->descriptor.pspgemm_csc->gemm->lda, request->descriptor.pspgemm_csc->gemm->ldb, request->descriptor.pspgemm_csc->gemm->ldc,
            request->descriptor.pspgemm_csc->packed_width,
            1, 0 != (LIBXSMM_GEMM_FLAG_BETA_0  & request->descriptor.pspgemm_csc->gemm->flags) ? 0 : 1,
            uid, nnz);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_PGEMMRMAC: { /* packed GEMM, B regular matrix, row-major */
      LIBXSMM_ASSERT(NULL != request->descriptor.pgemmacrm && 0 != request->descriptor.pgemmacrm->gemm);
      /* only floating point */
      if (LIBXSMM_GEMM_PRECISION_F64 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.pgemmacrm->gemm->datatype) ||
          LIBXSMM_GEMM_PRECISION_F32 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.pgemmacrm->gemm->datatype))
      {
        extra.nflops = 2 * request->descriptor.pgemmacrm->packed_width * request->descriptor.pgemmacrm->gemm->m * request->descriptor.pgemmacrm->gemm->n * request->descriptor.pgemmacrm->gemm->k;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_packed_gemm_ac_rm, &generated_code, request->descriptor.pgemmacrm->gemm, request->descriptor.pgemmacrm->packed_width);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid((libxsmm_gemm_prefetch_type)request->descriptor.pgemmacrm->gemm->prefetch);
          const char *const tname = libxsmm_typename((libxsmm_datatype)request->descriptor.pgemmacrm->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_w%u_a%i_b%i_p%i.pgemmacrm", target_arch, tname,
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.pgemmacrm->gemm->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.pgemmacrm->gemm->flags) ? 'n' : 't',
            request->descriptor.pgemmacrm->gemm->m,   request->descriptor.pgemmacrm->gemm->n,   request->descriptor.pgemmacrm->gemm->k,
            request->descriptor.pgemmacrm->gemm->lda, request->descriptor.pgemmacrm->gemm->ldb, request->descriptor.pgemmacrm->gemm->ldc,
            request->descriptor.pgemmacrm->packed_width,
            1, 0 != (LIBXSMM_GEMM_FLAG_BETA_0  & request->descriptor.pgemmacrm->gemm->flags) ? 0 : 1,
            uid);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_PGEMMRMBC: { /* packed GEMM, A regular matrix, row-major */
      LIBXSMM_ASSERT(NULL != request->descriptor.pgemmbcrm && 0 != request->descriptor.pgemmbcrm->gemm);
      /* only floating point */
      if (LIBXSMM_GEMM_PRECISION_F64 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.pgemmbcrm->gemm->datatype) ||
          LIBXSMM_GEMM_PRECISION_F32 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.pgemmbcrm->gemm->datatype))
      {
        extra.nflops = 2 * request->descriptor.pgemmbcrm->packed_width * request->descriptor.pgemmbcrm->gemm->m * request->descriptor.pgemmbcrm->gemm->n * request->descriptor.pgemmbcrm->gemm->k;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_packed_gemm_bc_rm, &generated_code, request->descriptor.pgemmbcrm->gemm, request->descriptor.pgemmbcrm->packed_width);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid((libxsmm_gemm_prefetch_type)request->descriptor.pgemmbcrm->gemm->prefetch);
          const char *const tname = libxsmm_typename((libxsmm_datatype)request->descriptor.pgemmbcrm->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_w%u_a%i_b%i_p%i.pgemmbcrm", target_arch, tname,
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.pgemmbcrm->gemm->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.pgemmbcrm->gemm->flags) ? 'n' : 't',
            request->descriptor.pgemmbcrm->gemm->m,   request->descriptor.pgemmbcrm->gemm->n,   request->descriptor.pgemmbcrm->gemm->k,
            request->descriptor.pgemmbcrm->gemm->lda, request->descriptor.pgemmbcrm->gemm->ldb, request->descriptor.pgemmbcrm->gemm->ldc,
            request->descriptor.pgemmbcrm->packed_width,
            1, 0 != (LIBXSMM_GEMM_FLAG_BETA_0  & request->descriptor.pgemmbcrm->gemm->flags) ? 0 : 1,
            uid);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_SREG: { /* sparse register kernel */
      LIBXSMM_ASSERT(NULL != request->descriptor.sreg && 0 != request->descriptor.sreg->gemm);
      LIBXSMM_ASSERT(NULL != request->descriptor.sreg->row_ptr && 0 != request->descriptor.sreg->column_idx && 0 != request->descriptor.sreg->values);
      /* only floating point */
      if (LIBXSMM_GEMM_PRECISION_F64 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.sreg->gemm->datatype) ||
          LIBXSMM_GEMM_PRECISION_F32 == /*LIBXSMM_GETENUM_OUT*/(request->descriptor.sreg->gemm->datatype))
      {
        const unsigned int nnz = request->descriptor.sreg->row_ptr[request->descriptor.sreg->gemm->m];
        extra.nflops = 2 * libxsmm_cpuid_vlen32(libxsmm_target_archid)/2 * request->descriptor.sreg->gemm->n * nnz;
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_spgemm_csr_reg_kernel, &generated_code, request->descriptor.sreg->gemm, target_arch,
          request->descriptor.sreg->row_ptr, request->descriptor.sreg->column_idx,
          (const double*)request->descriptor.sreg->values);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          const int uid = libxsmm_gemm_prefetch2uid((libxsmm_gemm_prefetch_type)request->descriptor.sreg->gemm->prefetch);
          const char *const tname = libxsmm_typename((libxsmm_datatype)request->descriptor.sreg->gemm->datatype);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_%s_%c%c_%ux%ux%u_%u_%u_%u_a%i_b%i_p%i.sreg", target_arch, tname,
            0 == (LIBXSMM_GEMM_FLAG_TRANS_A & request->descriptor.sreg->gemm->flags) ? 'n' : 't',
            0 == (LIBXSMM_GEMM_FLAG_TRANS_B & request->descriptor.sreg->gemm->flags) ? 'n' : 't',
            request->descriptor.sreg->gemm->m,   request->descriptor.sreg->gemm->n,   request->descriptor.sreg->gemm->k,
            request->descriptor.sreg->gemm->lda, request->descriptor.sreg->gemm->ldb, request->descriptor.sreg->gemm->ldc,
            1, 0 != (LIBXSMM_GEMM_FLAG_BETA_0  & request->descriptor.sreg->gemm->flags) ? 0 : 1,
            uid);
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_MELTW: { /* matcopy kernel */
      LIBXSMM_ASSERT(NULL != request->descriptor.meltw);
      {
        /* dispatch eltwise code with AVX512_BF16 by demoting seemlessly to the current CPU arch */
        if ( ( generated_code.arch >= LIBXSMM_X86_AVX512_SPR ) &&
             ( generated_code.arch <= LIBXSMM_X86_ALLFEAT )       ) {
          int emu_amx = 0;
          const char *const env_emu_amx = getenv("EMULATE_AMX");
          if ( 0 == env_emu_amx ) {
          } else {
            emu_amx = atoi(env_emu_amx);
          }
          if (emu_amx > 0) {
            generated_code.arch = libxsmm_cpuid();
          }
        }
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_mateltwise_kernel, &generated_code, request->descriptor.meltw);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          char tsizename[4];
          internal_get_typesize_string(tsizename, sizeof(tsizename), request->descriptor.meltw->datatype);
          /* adopt scheme which allows kernel names of LIBXSMM to appear in order (Intel VTune, etc.) */
          if ( request->descriptor.meltw->operation == LIBXSMM_MELTW_OPERATION_REDUCE_COLS_IDX ) {
            LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_tsize%s_idxtsize%u_%u_%ux%u_opcode%u_flags%u.meltw", target_arch, tsizename,
              request->descriptor.meltw->n, request->descriptor.meltw->m, request->descriptor.meltw->ldi, request->descriptor.meltw->ldo,
              (unsigned int)request->descriptor.meltw->operation, (unsigned int)request->descriptor.meltw->flags);
          } else {
            LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_tsize%s_%ux%u_%ux%u_opcode%u_flags%u_params%u.meltw", target_arch, tsizename,
              request->descriptor.meltw->m, request->descriptor.meltw->n, request->descriptor.meltw->ldi, request->descriptor.meltw->ldo,
              (unsigned int)request->descriptor.meltw->operation, (unsigned int)request->descriptor.meltw->flags, (unsigned int)request->descriptor.meltw->param);
          }
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_MEQN: { /* matequation kernel */
      LIBXSMM_ASSERT(NULL != request->descriptor.meltw);
      {
        /* dispatch eltwise code with AVX512_BF16 by demoting seemlessly to the current CPU arch */
        if ( ( generated_code.arch >= LIBXSMM_X86_AVX512_SPR ) &&
             ( generated_code.arch <= LIBXSMM_X86_ALLFEAT )       ) {
          int emu_amx = 0;
          const char *const env_emu_amx = getenv("EMULATE_AMX");
          if ( 0 == env_emu_amx ) {
          } else {
            emu_amx = atoi(env_emu_amx);
          }
          if (emu_amx > 0) {
            generated_code.arch = libxsmm_cpuid();
          }
        }
        LIBXSMM_NO_OFFLOAD(void, libxsmm_generator_matequation_kernel, &generated_code, request->descriptor.meqn);
# if !defined(LIBXSMM_VTUNE)
        if (0 > libxsmm_verbosity)
# endif
        {
          char tsizename[4];
          internal_get_typesize_string(tsizename, sizeof(tsizename), request->descriptor.meqn->datatype);
          LIBXSMM_SNPRINTF(jit_name, sizeof(jit_name), "libxsmm_%s_tsize%s_%ux%u_%u_eqn-idx%u.meltw", target_arch, tsizename,
            request->descriptor.meqn->m, request->descriptor.meqn->n, request->descriptor.meqn->ldo,
            (unsigned int)request->descriptor.meqn->eqn_idx );
        }
      }
    } break;
    case LIBXSMM_BUILD_KIND_USER: break;
# if !defined(NDEBUG) /* library code is expected to be mute */
    default: { /* unknown kind */
      static int error_once = 0;
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXSMM ERROR: invalid build request discovered!\n");
      }
      /*result = EXIT_FAILURE;*/
    }
# endif
  }

  if  (0 == generated_code.last_error /* no error raised */
    && 0 != generated_code.code_size /*check (tcopy issue?)*/)
  {
    char* code_buffer = NULL;
# if defined(__APPLE__) && defined(__arm64__)
# else
    void* code_buffer_result = &code_buffer;
# endif
    LIBXSMM_ASSERT(generated_code.code_size <= LIBXSMM_CODE_MAXSIZE);
    LIBXSMM_ASSERT(NULL != generated_code.generated_code);
    /* attempt to create executable buffer */
# if defined(__APPLE__) && defined(__arm64__)
    /* TODO: proper buffer x-allocation provides kernel info, etc. */
    code_buffer = (char*)mmap(0, generated_code.code_size, PROT_WRITE | PROT_EXEC | PROT_READ,
      MAP_PRIVATE | MAP_ANONYMOUS | MAP_JIT, -1, 0);
    result = ((0 <= (long long)code_buffer) ? EXIT_SUCCESS : EXIT_FAILURE);
# else
    result = libxsmm_xmalloc((void**)code_buffer_result, generated_code.code_size, 0/*auto*/,
      /* flag must be a superset of what's populated by libxsmm_malloc_attrib */
      LIBXSMM_MALLOC_FLAG_RWX, &extra, sizeof(extra));
# endif
    if (EXIT_SUCCESS == result) { /* check for success */
      LIBXSMM_ASSERT(NULL != code_buffer);
# if defined(__APPLE__) && defined(__arm64__)
      pthread_jit_write_protect_np(0/*false*/);
# endif
      /* copy temporary buffer into the prepared executable buffer */
# if defined(NDEBUG)
      { int i; /* precondition: jit_buffer == generated_code.generated_code */
        for (i = 0; i < (int)generated_code.code_size; ++i) code_buffer[i] = jit_buffer[i];
      }
# else
      memcpy(code_buffer, generated_code.generated_code, generated_code.code_size);
# endif
# if defined(__APPLE__) && defined(__arm64__)
      code->ptr = code_buffer; /* commit buffer */
      LIBXSMM_ASSERT(NULL != code->ptr && 0 == (LIBXSMM_CODE_STATIC & code->uval));
      sys_icache_invalidate(code_buffer, generated_code.code_size);
      pthread_jit_write_protect_np(1/*true*/);
# else
      /* attribute/protect buffer and revoke unnecessary flags */
      result = libxsmm_malloc_attrib((void**)code_buffer_result, LIBXSMM_MALLOC_FLAG_X, jit_name);
      if (EXIT_SUCCESS == result) { /* check for success */
        code->ptr = code_buffer; /* commit buffer */
        LIBXSMM_ASSERT(NULL != code->ptr && 0 == (LIBXSMM_CODE_STATIC & code->uval));
#   if defined(__aarch64__)
#     if defined(__clang__)
        __clear_cache(code_buffer, code_buffer + generated_code.code_size);
#     else
        __builtin___clear_cache(code_buffer, code_buffer + generated_code.code_size);
#     endif
#   endif
      }
      else { /* release buffer */
        libxsmm_xfree(code_buffer, 0/*no check*/);
      }
# endif
    }
  }
  else if (request->kind == LIBXSMM_BUILD_KIND_USER && NULL != request->descriptor.ptr) { /* user-data */
    if (0 != request->user_size) {
      void* user_data = &code->ptr;
      result = libxsmm_xmalloc((void**)user_data, request->user_size, 0/*auto*/,
        LIBXSMM_MALLOC_FLAG_PRIVATE, &extra, sizeof(extra));
    }
    else {
      result = EXIT_SUCCESS;
      code->ptr = NULL;
    }
  }
  else {
    result = (0 != generated_code.last_error ? generated_code.last_error : EXIT_FAILURE);
  }
#else /* unsupported platform */
  LIBXSMM_UNUSED(request); LIBXSMM_UNUSED(regindex); LIBXSMM_UNUSED(code);
  /* libxsmm_get_target_arch also serves as a runtime check whether JIT is available or not */
  if (LIBXSMM_X86_GENERIC <= libxsmm_target_archid) result = EXIT_FAILURE;
#endif
  return result;
}


LIBXSMM_API_INLINE void internal_pad_descriptor(libxsmm_descriptor* desc, signed char size)
{
  LIBXSMM_ASSERT(LIBXSMM_DESCRIPTOR_MAXSIZE < 128 && NULL != desc);
  LIBXSMM_ASSERT(LIBXSMM_DIFF_SIZE <= LIBXSMM_DESCRIPTOR_MAXSIZE);
  LIBXSMM_ASSERT(LIBXSMM_HASH_SIZE <= LIBXSMM_DIFF_SIZE);
  for (; size < LIBXSMM_DIFF_SIZE; ++size) desc->data[size] = 0;
}


LIBXSMM_API_INLINE libxsmm_code_pointer internal_find_code(libxsmm_descriptor* desc, size_t desc_size, size_t user_size, unsigned int* hash)
{
  libxsmm_code_pointer flux_entry = { 0 };
  const int is_big_desc = LIBXSMM_DESCRIPTOR_ISBIG(desc->kind);
  const signed char size = (signed char)(sizeof(libxsmm_descriptor_kind) + desc_size);
  LIBXSMM_DIFF_DECL(LIBXSMM_DIFF_SIZE, xdesc);
#if !defined(NDEBUG) && (0 != LIBXSMM_JIT)
  int build = EXIT_SUCCESS;
#endif
#if defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
# if defined(LIBXSMM_NTHREADS_USE)
  const unsigned int tid = libxsmm_get_tid();
  internal_cache_type *const cache = internal_cache_buffer + tid;
# else
  static LIBXSMM_TLS internal_cache_type internal_cache_buffer;
  internal_cache_type *const cache = &internal_cache_buffer;
# endif
  unsigned char cache_index;
  const unsigned int ninit = LIBXSMM_ATOMIC_LOAD(&libxsmm_ninit, LIBXSMM_ATOMIC_RELAXED);
  internal_pad_descriptor(desc, size);
  LIBXSMM_ASSERT(NULL != hash);
  if (0 == is_big_desc) {
    LIBXSMM_DIFF_LOAD(LIBXSMM_DIFF_SIZE, xdesc, desc);
    LIBXSMM_DIFF_N(unsigned char, cache_index, LIBXSMM_DIFF(LIBXSMM_DIFF_SIZE), xdesc, cache->entry.keys,
      LIBXSMM_DIFF_SIZE, LIBXSMM_CACHE_STRIDE, cache->entry.hit, cache->entry.size);
  }
  else {
    cache_index = (unsigned char)libxsmm_diff_n(desc, cache->entry.keys,
      size, LIBXSMM_CACHE_STRIDE, cache->entry.hit, cache->entry.size);
  }
  if (ninit == cache->entry.id && cache_index < cache->entry.size) { /* valid hit */
    flux_entry = cache->entry.code[cache_index];
    cache->entry.hit = cache_index;
  }
  else
#else
  internal_pad_descriptor(desc, size);
  LIBXSMM_ASSERT(NULL != hash);
#endif
  {
    unsigned int i, i0, mode = 0, diff = 1;
    *hash = LIBXSMM_CRC32(LIBXSMM_HASH_SIZE)(LIBXSMM_HASH_SEED, desc);
    i0 = i = LIBXSMM_MOD2(*hash, LIBXSMM_CAPACITY_REGISTRY);
    LIBXSMM_ASSERT(&desc->kind == &desc->gemm.pad && desc->kind == desc->gemm.pad);
    LIBXSMM_ASSERT(NULL != internal_registry);
    do { /* use calculated location and check if the requested code is already JITted */
#if (1 < INTERNAL_REGLOCK_MAXN) || !LIBXSMM_LOCK_TYPE_ISRW(LIBXSMM_REGLOCK) /* read registered code */
# if 1 /* omitting an atomic load is safe but avoids race-detectors to highlight this location */
      uintptr_t *const fluxaddr = &internal_registry[i].uval;
      flux_entry.uval = LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_LOAD, LIBXSMM_BITS)(fluxaddr, LIBXSMM_ATOMIC_RELAXED);
# else
      flux_entry = internal_registry[i];
# endif
#else
      LIBXSMM_LOCK_ACQREAD(LIBXSMM_REGLOCK, internal_reglock_ptr);
      flux_entry = internal_registry[i]; /* read registered code */
      LIBXSMM_LOCK_RELREAD(LIBXSMM_REGLOCK, internal_reglock_ptr);
#endif
      if ((NULL != flux_entry.ptr_const || 1 == mode) && 2 > mode) { /* confirm entry */
        if (NULL != flux_entry.ptr_const) {
          if (0 == is_big_desc) {
#if !defined(LIBXSMM_CACHE_MAXSIZE) || (0 == (LIBXSMM_CACHE_MAXSIZE))
            LIBXSMM_DIFF_LOAD(LIBXSMM_DIFF_SIZE, xdesc, desc);
#endif
            diff = LIBXSMM_DIFF(LIBXSMM_DIFF_SIZE)(xdesc, internal_registry_keys + i, 0/*dummy*/);
          }
          else {
            diff = libxsmm_diff(desc, internal_registry_keys + i, size);
          }
        }
#if !defined(NDEBUG)
        else LIBXSMM_ASSERT(0 != diff);
#endif
        if (0 != diff) { /* search for code version */
          if (0 == mode) { /* transition to higher mode */
            i0 = i; /* keep current position on record */
#if defined(LIBXSMM_HASH_COLLISION)
            /* enter code generation, and collision fix-up */
            if (0 == (LIBXSMM_HASH_COLLISION & flux_entry.uval)) {
              LIBXSMM_ASSERT(NULL != flux_entry.ptr_const); /* collision */
              mode = 3;
            }
            else
#endif      /* search for an existing code version */
            mode = 1; /* else */
          }
          i = LIBXSMM_MOD2(i + 1, LIBXSMM_CAPACITY_REGISTRY);
          if (i == i0) { /* search finished, no code version exists */
#if defined(LIBXSMM_HASH_COLLISION)
            mode = 3; /* enter code generation, and collision fix-up */
#else
            mode = 2; /* enter code generation */
#endif
            if (LIBXSMM_KERNEL_KIND_MATMUL == LIBXSMM_DESCRIPTOR_KIND(desc->kind)) {
              internal_update_mmstatistic(&desc->gemm.desc, 0, 1/*collision*/, 0, 0);
            }
          }
          LIBXSMM_ASSERT(0 != diff); /* continue */
        }
      }
      else { /* enter code generation (there is no code version yet) */
        LIBXSMM_ASSERT(0 == mode || 1 < mode);
#if (0 == LIBXSMM_JIT)
        LIBXSMM_UNUSED(user_size);
#else
        if (LIBXSMM_X86_GENERIC <= libxsmm_target_archid || /* check if JIT is supported (CPUID) */
           (LIBXSMM_KERNEL_KIND_USER == LIBXSMM_DESCRIPTOR_KIND(desc->kind)))
        {
          LIBXSMM_ASSERT(0 != mode || NULL == flux_entry.ptr_const/*code version does not exist*/);
          INTERNAL_FIND_CODE_LOCK(lock, i, diff, flux_entry.ptr); /* lock the registry entry */
          if (NULL == internal_registry[i].ptr_const) { /* double-check registry after acquiring the lock */
            libxsmm_build_request request; /* setup the code build request */
            LIBXSMM_ASSERT(LIBXSMM_KERNEL_UNREGISTERED > LIBXSMM_DESCRIPTOR_KIND(desc->kind));
            request.kind = (libxsmm_build_kind)LIBXSMM_DESCRIPTOR_KIND(desc->kind);
            request.descriptor.ptr = &desc->gemm.desc;
            request.user_size = user_size;
# if defined(NDEBUG)
            if (EXIT_SUCCESS == libxsmm_build(&request, i, &flux_entry) && NULL != flux_entry.ptr_const)
# else
            build = libxsmm_build(&request, i, &flux_entry);
            if (EXIT_SUCCESS == build && NULL != flux_entry.ptr_const)
# endif
            {
              LIBXSMM_ASSIGN127(internal_registry_keys + i, desc);
# if (1 < INTERNAL_REGLOCK_MAXN)
              LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, LIBXSMM_BITS)(&internal_registry[i].ptr, flux_entry.ptr, LIBXSMM_ATOMIC_SEQ_CST);
# else
              internal_registry[i] = flux_entry;
# endif
# if defined(LIBXSMM_HASH_COLLISION)
              if (2 < mode) { /* arrived from collision state; now mark as collision */
                libxsmm_code_pointer fix_entry;
#   if (1 < INTERNAL_REGLOCK_MAXN)
                fix_entry.ptr = LIBXSMM_ATOMIC_LOAD(&internal_registry[i0].ptr, LIBXSMM_ATOMIC_RELAXED);
#   else
                fix_entry = internal_registry[i0];
#   endif
                LIBXSMM_ASSERT(NULL != fix_entry.ptr_const);
                if (0 == (LIBXSMM_HASH_COLLISION & fix_entry.uval)) {
                  fix_entry.uval |= LIBXSMM_HASH_COLLISION; /* mark current entry as collision */
#   if (1 < INTERNAL_REGLOCK_MAXN)
                  LIBXSMM_ATOMIC_STORE(&internal_registry[i0].ptr, fix_entry.ptr, LIBXSMM_ATOMIC_RELAXED);
#   else
                  internal_registry[i0] = fix_entry;
#   endif
                }
              }
# endif
            }
            if (LIBXSMM_KERNEL_KIND_MATMUL == LIBXSMM_DESCRIPTOR_KIND(desc->kind)) {
              internal_update_mmstatistic(&desc->gemm.desc, 1/*try*/, 0, 0, 0);
            }
            /* leave here even in case of a build-error; do not use break (inside of locked region) */
            diff = 0;
          }
          INTERNAL_FIND_CODE_UNLOCK(lock);
          if (0 != diff) { /* acquire registry slot */
            if (0 == mode) { /* initial condition */
              mode = 2; /* continue to linearly search for an empty slot */
              i0 = i; /* keep current position on record */
            }
            do { /* continue to linearly search for an available slot */
              i = LIBXSMM_MOD2(i + 1, LIBXSMM_CAPACITY_REGISTRY);
              if (NULL == internal_registry[i].ptr_const) break;
            } while (i != i0);
            if (i == i0) { /* out of capacity (no registry slot available) */
              diff = 0; /* do not use break if inside of locked region */
            }
            flux_entry.ptr = NULL; /* no result */
          }
        }
        else /* JIT-code generation not available */
#endif
        { /* leave the dispatch loop */
          if (LIBXSMM_KERNEL_KIND_MATMUL == LIBXSMM_DESCRIPTOR_KIND(desc->kind)) {
            internal_update_mmstatistic(&desc->gemm.desc, 1/*try*/, 0, 0, 0);
          }
#if !defined(NDEBUG) && (0 != LIBXSMM_JIT)
          build = EXIT_FAILURE;
#endif
          flux_entry.ptr = NULL;
          diff = 0;
        }
      }
    } while (0 != diff);
#if defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
    if (NULL != flux_entry.ptr_const) { /* keep code version on record (cache) */
      LIBXSMM_ASSERT(0 == diff);
      if (ninit == cache->entry.id) { /* maintain cache */
        if (cache->entry.size < internal_cache_size) { /* grow */
          INTERNAL_FIND_CODE_CACHE_GROW(cache_index, cache->entry.size);
          LIBXSMM_ASSERT(cache->entry.size <= internal_cache_size);
        }
        else { /* evict */
          LIBXSMM_ASSERT(cache->entry.hit < cache->entry.size);
          INTERNAL_FIND_CODE_CACHE_EVICT(cache_index, cache->entry.size, cache->entry.hit);
        }
      }
      else if (0 != internal_cache_size) { /* reset cache */
        /* INTERNAL_FIND_CODE_CACHE_GROW doubles size (and would expose invalid entries) */
        memset(cache->entry.keys, 0, LIBXSMM_CACHE_MAXSIZE * sizeof(*cache->entry.keys));
        cache->entry.id = ninit;
        cache->entry.size = 1;
        cache_index = 0;
      }
      LIBXSMM_MEMCPY127(cache->entry.keys + cache_index, desc, 0 == is_big_desc ? LIBXSMM_DIFF_SIZE : size);
      cache->entry.code[cache_index] = flux_entry;
      cache->entry.hit = cache_index;
    }
# if !defined(NDEBUG)
    else {
      memset(cache, 0, sizeof(*cache));
    }
# endif
#endif
  }
#if defined(LIBXSMM_HASH_COLLISION)
  flux_entry.uval &= ~(LIBXSMM_CODE_STATIC | LIBXSMM_HASH_COLLISION); /* clear non-JIT and collision flag */
#else
  flux_entry.uval &= ~LIBXSMM_CODE_STATIC; /* clear non-JIT flag */
#endif
#if (0 != LIBXSMM_JIT)
  assert( /*!LIBXSMM_ASSERT*/
    LIBXSMM_KERNEL_KIND_MATMUL != LIBXSMM_DESCRIPTOR_KIND(desc->kind)
    || NULL != flux_entry.ptr_const
    || 1 == internal_reglock_count
    || EXIT_SUCCESS != build);
#endif
  return flux_entry;
}


LIBXSMM_API_INTERN const libxsmm_kernel_xinfo* libxsmm_get_kernel_xinfo(libxsmm_code_pointer code,
  const libxsmm_descriptor** desc, size_t* code_size)
{
  libxsmm_kernel_xinfo* result = NULL;
  void *const result_address = &result;
  int flags = LIBXSMM_MALLOC_FLAG_X;
  if (NULL != code.ptr_const && EXIT_SUCCESS == libxsmm_get_malloc_xinfo(
    code.ptr_const, code_size, &flags, (void**)result_address) && NULL != result)
  {
    if (NULL != desc) {
      if (NULL != internal_registry && NULL != internal_registry_keys && result->registered < (LIBXSMM_CAPACITY_REGISTRY)
#if defined(LIBXSMM_HASH_COLLISION)
        && code.uval == (~LIBXSMM_HASH_COLLISION & internal_registry[result->registered].uval)
#else
        && code.ptr_const == internal_registry[result->registered].ptr_const
#endif
        && LIBXSMM_KERNEL_UNREGISTERED > LIBXSMM_DESCRIPTOR_KIND(internal_registry_keys[result->registered].entry.kind))
      {
        *desc = &internal_registry_keys[result->registered].entry;
      }
      else *desc = NULL;
    }
  }
  else {
    LIBXSMM_ASSERT(NULL == result);
    if (NULL != code_size) *code_size = 0;
    if (NULL != desc) *desc = NULL;
  }
  return result;
}


LIBXSMM_API int libxsmm_get_kernel_info(const void* kernel, libxsmm_kernel_info* info)
{
  int result;
  const libxsmm_kernel_xinfo* xinfo;
  libxsmm_kernel_info result_info;
  const libxsmm_descriptor* desc;
  libxsmm_code_pointer code;
  code.ptr_const = kernel;
  LIBXSMM_MEMZERO127(&result_info);
  xinfo = libxsmm_get_kernel_xinfo(code, &desc, &result_info.code_size);
  if (NULL != xinfo) {
    if (NULL != desc) {
      const libxsmm_kernel_kind kind = (libxsmm_kernel_kind)LIBXSMM_DESCRIPTOR_KIND(desc->kind);
      result_info.kind = kind;
      if (LIBXSMM_KERNEL_KIND_USER == kind) {
        result_info.code_size = 0; /* invalid */
      }
    }
    else {
      result_info.kind = LIBXSMM_KERNEL_UNREGISTERED;
    }
    result_info.nflops = xinfo->nflops;
    LIBXSMM_ASSIGN127(info, &result_info);
    result = EXIT_SUCCESS;
  }
  else {
    LIBXSMM_ASSERT(NULL == desc);
    if (NULL != info) {
      LIBXSMM_ASSIGN127(info, &result_info);
      result = EXIT_FAILURE;
    }
    else {
      result = EXIT_SUCCESS;
    }
  }
  return result;
}


LIBXSMM_API int libxsmm_get_mmkernel_info(libxsmm_xmmfunction kernel, libxsmm_mmkernel_info* info)
{
  libxsmm_code_pointer code;
  static int error_once = 0;
  int result;
  code.xgemm = kernel;
  if (NULL != info) {
    const libxsmm_descriptor* desc;
    if (NULL != libxsmm_get_kernel_xinfo(code, &desc, NULL/*code_size*/) &&
        NULL != desc && LIBXSMM_KERNEL_KIND_MATMUL == LIBXSMM_DESCRIPTOR_KIND(desc->kind))
    {
      info->iprecision = (libxsmm_gemm_precision)LIBXSMM_GETENUM_INP(desc->gemm.desc.datatype);
      info->oprecision = (libxsmm_gemm_precision)LIBXSMM_GETENUM_OUT(desc->gemm.desc.datatype);
      info->prefetch = (libxsmm_gemm_prefetch_type)desc->gemm.desc.prefetch;
      info->flags = desc->gemm.desc.flags;
      info->lda = desc->gemm.desc.lda;
      info->ldb = desc->gemm.desc.ldb;
      info->ldc = desc->gemm.desc.ldc;
      info->m = desc->gemm.desc.m;
      info->n = desc->gemm.desc.n;
      info->k = desc->gemm.desc.k;
      result = EXIT_SUCCESS;
    }
    else {
#if defined(__APPLE__) && defined(__arm64__)
      /* TODO: proper buffer x-allocation provides kernel info, etc. */
      info->iprecision = LIBXSMM_GEMM_PRECISION_F32;
      info->oprecision = LIBXSMM_GEMM_PRECISION_F32;
      info->prefetch = LIBXSMM_GEMM_PREFETCH_SIGONLY;
      info->flags = LIBXSMM_GEMM_FLAG_NONE;
      info->lda = info->ldb = info->ldc = 1;
      info->m = info->n = info->k = 1;
      result = EXIT_SUCCESS;
#else
      if ( 0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        if (NULL == code.ptr_const) {
          fprintf(stderr, "LIBXSMM ERROR: NULL-kernel cannot be inspected!\n");
        }
        else {
          fprintf(stderr, "LIBXSMM ERROR: invalid kernel cannot be inspected!\n");
        }
      }
      result = EXIT_FAILURE;
#endif
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: invalid argument!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_get_meltwkernel_info(libxsmm_xmeltwfunction kernel, libxsmm_meltwkernel_info* info)
{
  libxsmm_code_pointer code;
  static int error_once = 0;
  int result;
  code.xmateltw = kernel;
  if (NULL != info) {
    const libxsmm_descriptor* desc;
    if (NULL != libxsmm_get_kernel_xinfo(code, &desc, NULL/*code_size*/) &&
        NULL != desc && LIBXSMM_KERNEL_KIND_MELTW == LIBXSMM_DESCRIPTOR_KIND(desc->kind))
    {
      info->datatype = desc->meltw.desc.datatype;
      info->operation = desc->meltw.desc.operation;
      info->flags = desc->meltw.desc.flags;
      info->ldi = desc->meltw.desc.ldi;
      info->ldo = desc->meltw.desc.ldo;
      info->m = desc->meltw.desc.m;
      info->n = desc->meltw.desc.n;
      result = EXIT_SUCCESS;
    }
    else {
      if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: invalid kernel cannot be inspected!\n");
      }
      result = EXIT_FAILURE;
    }
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: invalid argument!\n");
    }
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_get_registry_info(libxsmm_registry_info* info)
{
  int result = EXIT_SUCCESS;
  LIBXSMM_INIT /* verbosity */
  if (0 != info && 0 != internal_registry) {
    size_t i;
    LIBXSMM_MEMZERO127(info); /* info->nstatic = 0; info->size = 0; */
    info->nbytes = (LIBXSMM_CAPACITY_REGISTRY) * (sizeof(libxsmm_code_pointer) + sizeof(libxsmm_descriptor));
    info->capacity = LIBXSMM_CAPACITY_REGISTRY;
#if defined(LIBXSMM_CACHE_MAXSIZE) && (0 < (LIBXSMM_CACHE_MAXSIZE))
    info->ncache = internal_cache_size;
#else
    info->ncache = 0;
#endif
    for (i = 0; i < (LIBXSMM_CAPACITY_REGISTRY); ++i) {
      libxsmm_code_pointer code = internal_registry[i];
      if (0 != code.ptr_const && EXIT_SUCCESS == result) {
        if (0 == (LIBXSMM_CODE_STATIC & code.uval)) { /* check for allocated/generated JIT-code */
          size_t buffer_size = 0;
          void* buffer = 0;
#if defined(LIBXSMM_HASH_COLLISION)
          code.uval &= ~LIBXSMM_HASH_COLLISION; /* clear collision flag */
#endif
          result = libxsmm_get_malloc_xinfo(code.ptr_const, &buffer_size, NULL/*flags*/, &buffer);
          if (EXIT_SUCCESS == result) {
            info->nbytes += LIBXSMM_UP2(buffer_size + (((char*)code.ptr_const) - (char*)buffer), LIBXSMM_PAGE_MINSIZE);
          }
        }
        else {
          ++info->nstatic;
        }
        ++info->size;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_INLINE void* internal_get_registry_entry(int i, libxsmm_kernel_kind kind, const void** key)
{
  void* result = NULL;
  LIBXSMM_ASSERT(kind < LIBXSMM_KERNEL_UNREGISTERED && NULL != internal_registry);
  for (; i < (LIBXSMM_CAPACITY_REGISTRY); ++i) {
    const libxsmm_code_pointer regentry = internal_registry[i];
    if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(regentry.ptr_const,
      NULL/*code_size*/, NULL/*flags*/, &result) && NULL != result)
    {
      const libxsmm_kernel_xinfo info = *(const libxsmm_kernel_xinfo*)result;
      const libxsmm_descriptor *const desc = &internal_registry_keys[info.registered].entry;
      if (LIBXSMM_DESCRIPTOR_KIND(desc->kind) == (int)kind) {
        if (NULL != key) *key = desc->user.desc;
        result = regentry.ptr;
        break;
      }
    }
  }
  return result;
}


LIBXSMM_API void* libxsmm_get_registry_begin(libxsmm_kernel_kind kind, const void** key)
{
  void* result = NULL;
  if (kind < LIBXSMM_KERNEL_UNREGISTERED && NULL != internal_registry) {
    result = internal_get_registry_entry(0, kind, key);
  }
  return result;
}


LIBXSMM_API void* libxsmm_get_registry_next(const void* regentry, const void** key)
{
  void* result = NULL;
  const libxsmm_descriptor* desc;
  libxsmm_code_pointer entry;
  entry.ptr_const = regentry;
  if (NULL != libxsmm_get_kernel_xinfo(entry, &desc, NULL/*code_size*/)
    /* given regentry is indeed a registered kernel */
    && NULL != desc)
  {
    result = internal_get_registry_entry(
      (int)(desc - &internal_registry_keys->entry + 1),
      (libxsmm_kernel_kind)LIBXSMM_DESCRIPTOR_KIND(desc->kind), key);
  }
  return result;
}


LIBXSMM_API void* libxsmm_xregister(const void* key, size_t key_size,
  size_t value_size, const void* value_init, unsigned int* key_hash)
{
  static int error_once = 0;
  void* result;
  LIBXSMM_INIT /* verbosity */
  if (NULL != key && 0 < key_size && LIBXSMM_DESCRIPTOR_MAXSIZE >= key_size) {
    libxsmm_descriptor wrap;
    unsigned int hash = 0;
    void* dst;
#if defined(LIBXSMM_UNPACKED) /* CCE/Classic */
    LIBXSMM_MEMSET127(&wrap, 0, key_size);
#endif
    LIBXSMM_MEMCPY127(wrap.user.desc, key, key_size);
    wrap.kind = (libxsmm_descriptor_kind)(LIBXSMM_DESCRIPTOR_SIGSIZE >= key_size
      ? ((libxsmm_descriptor_kind)LIBXSMM_KERNEL_KIND_USER)
      : LIBXSMM_DESCRIPTOR_BIG(LIBXSMM_KERNEL_KIND_USER));
    dst = internal_find_code(&wrap, key_size, value_size, &hash).ptr;
    if (NULL != key_hash) *key_hash = hash;
    if (NULL != dst) {
      size_t size;
      if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(dst, &size, NULL/*flags*/, NULL/*extra*/)
        && value_size <= size)
      {
        if (NULL != value_init) memcpy(dst, value_init, value_size);
        result = dst;
      }
      else {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: value too large for previously registered key!\n");
        }
        result = NULL;
      }
    }
    else result = NULL;
  }
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      if (LIBXSMM_DESCRIPTOR_MAXSIZE >= key_size) {
        fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xregister specified!\n");
      }
      else {
        fprintf(stderr, "LIBXSMM ERROR: libxsmm_xregister has maximum key-size of %i Byte!\n",
          LIBXSMM_DESCRIPTOR_MAXSIZE);
      }
    }
    result = NULL;
  }
  return result;
}


LIBXSMM_API void* libxsmm_xdispatch(const void* key, size_t key_size, unsigned int* key_hash)
{
  void* result;
  LIBXSMM_INIT /* verbosity */
#if !defined(NDEBUG)
  if (NULL != key && 0 < key_size && LIBXSMM_DESCRIPTOR_MAXSIZE >= key_size)
#endif
  {
    unsigned int hash = 0;
    libxsmm_descriptor wrap;
#if defined(LIBXSMM_UNPACKED) /* CCE/Classic */
    LIBXSMM_MEMSET127(&wrap, 0, key_size);
#endif
    LIBXSMM_MEMCPY127(wrap.user.desc, key, key_size);
    wrap.kind = (libxsmm_descriptor_kind)(LIBXSMM_DESCRIPTOR_SIGSIZE >= key_size
      ? ((libxsmm_descriptor_kind)LIBXSMM_KERNEL_KIND_USER)
      : LIBXSMM_DESCRIPTOR_BIG(LIBXSMM_KERNEL_KIND_USER));
    result = internal_find_code(&wrap, key_size, 0/*user_size*/, &hash).ptr;
    if (NULL != key_hash) *key_hash = hash;
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xdispatch specified!\n");
    }
    result = NULL;
  }
#endif
  return result;
}


LIBXSMM_API void libxsmm_xrelease(const void* key, size_t key_size)
{
  libxsmm_release_kernel(libxsmm_xdispatch(key, key_size, NULL/*key_hash*/));
}


LIBXSMM_API libxsmm_xmmfunction libxsmm_xmmdispatch(const libxsmm_gemm_descriptor* descriptor)
{
  libxsmm_xmmfunction result;
  LIBXSMM_INIT /* verbosity */
#if !defined(LIBXSMM_UNPACKED) /* CCE/Classic */
  LIBXSMM_ASSERT((sizeof(*descriptor) + sizeof(libxsmm_descriptor_kind)) <= (LIBXSMM_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    const int batch_reduce =
      LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS |
      LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET |
      LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE;
    libxsmm_descriptor wrap;
#if defined(LIBXSMM_UNPACKED) /* CCE/Classic */
    LIBXSMM_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
    LIBXSMM_ASSIGN127(&wrap.gemm.desc, descriptor);
    wrap.kind = (libxsmm_descriptor_kind)(0 == (batch_reduce & descriptor->flags)
      ? ((libxsmm_descriptor_kind)LIBXSMM_KERNEL_KIND_MATMUL)
      : LIBXSMM_DESCRIPTOR_BIG(LIBXSMM_KERNEL_KIND_MATMUL));
    if (0 != (0x80 & descriptor->prefetch)) { /* "sign"-bit of byte-value is set */
      wrap.gemm.desc.prefetch = (unsigned char)libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
    }
    result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xgemm;
#if defined(_DEBUG)
    if (LIBXSMM_VERBOSITY_HIGH <= libxsmm_verbosity && INT_MAX != libxsmm_verbosity && NULL != result.xmm) {
      LIBXSMM_STDIO_ACQUIRE();
      fprintf(stderr, "\nLIBXSMM: ");
      libxsmm_gemm_xprint(stderr, result, NULL/*a*/, NULL/*b*/, NULL/*c*/);
      LIBXSMM_STDIO_RELEASE();
    }
#endif
  }
  else { /* quietly accept NULL-descriptor */
    result.xmm = NULL;
  }
  return result;
}


LIBXSMM_API libxsmm_dmmfunction libxsmm_dmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.dmm;
}


LIBXSMM_API libxsmm_smmfunction libxsmm_smmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.smm;
}


LIBXSMM_API libxsmm_bsmmfunction libxsmm_bsmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.bsmm;
}


LIBXSMM_API libxsmm_bmmfunction libxsmm_bmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.bmm;
}


LIBXSMM_API libxsmm_wimmfunction libxsmm_wimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.wimm;
}


LIBXSMM_API libxsmm_ssbimmfunction libxsmm_ssbimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.ssbimm;
}


LIBXSMM_API libxsmm_usbimmfunction libxsmm_usbimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_A_UNSIGNED, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.usbimm;
}


LIBXSMM_API libxsmm_subimmfunction libxsmm_subimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.subimm;
}


LIBXSMM_API libxsmm_uubimmfunction libxsmm_uubimmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_AB_UNSIGNED, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.uubimm;
}


LIBXSMM_API libxsmm_sububmmfunction libxsmm_sububmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_C_UNSIGNED, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.sububmm;
}


LIBXSMM_API libxsmm_dmmfunction_reducebatch_addr libxsmm_dmmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.dmra;
}


LIBXSMM_API libxsmm_smmfunction_reducebatch_addr libxsmm_smmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.smra;
}


LIBXSMM_API libxsmm_bsmmfunction_reducebatch_addr libxsmm_bsmmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.bsmra;
}


LIBXSMM_API libxsmm_bmmfunction_reducebatch_addr libxsmm_bmmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.bmra;
}


LIBXSMM_API libxsmm_wimmfunction_reducebatch_addr libxsmm_wimmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.wimra;
}


LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_addr libxsmm_ssbimmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.ssbimra;
}


LIBXSMM_API libxsmm_usbimmfunction_reducebatch_addr libxsmm_usbimmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.usbimra;
}


LIBXSMM_API libxsmm_subimmfunction_reducebatch_addr libxsmm_subimmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.subimra;
}


LIBXSMM_API libxsmm_uubimmfunction_reducebatch_addr libxsmm_uubimmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_AB_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.uubimra;
}


LIBXSMM_API libxsmm_sububmmfunction_reducebatch_addr libxsmm_sububmmdispatch_reducebatch_addr(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_C_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.sububmra;
}


LIBXSMM_API libxsmm_dmmfunction_reducebatch_addr libxsmm_dmmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.dmra;
}


LIBXSMM_API libxsmm_smmfunction_reducebatch_addr libxsmm_smmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.smra;
}


LIBXSMM_API libxsmm_bsmmfunction_reducebatch_addr libxsmm_bsmmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.bsmra;
}


LIBXSMM_API libxsmm_bmmfunction_reducebatch_addr libxsmm_bmmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.bmra;
}


LIBXSMM_API libxsmm_wimmfunction_reducebatch_addr libxsmm_wimmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.wimra;
}


LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_addr libxsmm_ssbimmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.ssbimra;
}


LIBXSMM_API libxsmm_usbimmfunction_reducebatch_addr libxsmm_usbimmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.usbimra;
}


LIBXSMM_API libxsmm_subimmfunction_reducebatch_addr libxsmm_subimmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.subimra;
}


LIBXSMM_API libxsmm_uubimmfunction_reducebatch_addr libxsmm_uubimmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_AB_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.uubimra;
}


LIBXSMM_API libxsmm_sububmmfunction_reducebatch_addr libxsmm_sububmmdispatch_reducebatch_addr_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_C_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.sububmra;
}


LIBXSMM_API libxsmm_dmmfunction_reducebatch_offs libxsmm_dmmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.dmro;
}


LIBXSMM_API libxsmm_smmfunction_reducebatch_offs libxsmm_smmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.smro;
}


LIBXSMM_API libxsmm_bsmmfunction_reducebatch_offs libxsmm_bsmmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.bsmro;
}


LIBXSMM_API libxsmm_bmmfunction_reducebatch_offs libxsmm_bmmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.bmro;
}


LIBXSMM_API libxsmm_wimmfunction_reducebatch_offs libxsmm_wimmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.wimro;
}


LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_offs libxsmm_ssbimmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.ssbimro;
}


LIBXSMM_API libxsmm_usbimmfunction_reducebatch_offs libxsmm_usbimmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.usbimro;
}


LIBXSMM_API libxsmm_subimmfunction_reducebatch_offs libxsmm_subimmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.subimro;
}


LIBXSMM_API libxsmm_uubimmfunction_reducebatch_offs libxsmm_uubimmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_AB_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.uubimro;
}


LIBXSMM_API libxsmm_sububmmfunction_reducebatch_offs libxsmm_sububmmdispatch_reducebatch_offs(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_C_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result = libxsmm_xmmdispatch(desc);
  return result.sububmro;
}


LIBXSMM_API libxsmm_dmmfunction_reducebatch_offs libxsmm_dmmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.dmro;
}


LIBXSMM_API libxsmm_smmfunction_reducebatch_offs libxsmm_smmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.smro;
}


LIBXSMM_API libxsmm_bsmmfunction_reducebatch_offs libxsmm_bsmmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.bsmro;
}


LIBXSMM_API libxsmm_bmmfunction_reducebatch_offs libxsmm_bmmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.bmro;
}


LIBXSMM_API libxsmm_wimmfunction_reducebatch_offs libxsmm_wimmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.wimro;
}


LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_offs libxsmm_ssbimmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.ssbimro;
}


LIBXSMM_API libxsmm_usbimmfunction_reducebatch_offs libxsmm_usbimmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.usbimro;
}


LIBXSMM_API libxsmm_subimmfunction_reducebatch_offs libxsmm_subimmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.subimro;
}


LIBXSMM_API libxsmm_uubimmfunction_reducebatch_offs libxsmm_uubimmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_AB_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.uubimro;
}


LIBXSMM_API libxsmm_sububmmfunction_reducebatch_offs libxsmm_sububmmdispatch_reducebatch_offs_unroll(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_C_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  result = libxsmm_xmmdispatch(desc);
  return result.sububmro;
}


LIBXSMM_API libxsmm_dmmfunction_reducebatch_strd libxsmm_dmmdispatch_reducebatch_strd(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.dmrs;
}


LIBXSMM_API libxsmm_smmfunction_reducebatch_strd libxsmm_smmdispatch_reducebatch_strd(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.smrs;
}


LIBXSMM_API libxsmm_bsmmfunction_reducebatch_strd libxsmm_bsmmdispatch_reducebatch_strd(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.bsmrs;
}


LIBXSMM_API libxsmm_bmmfunction_reducebatch_strd libxsmm_bmmdispatch_reducebatch_strd(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.bmrs;
}


LIBXSMM_API libxsmm_wimmfunction_reducebatch_strd libxsmm_wimmdispatch_reducebatch_strd(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.wimrs;
}


LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_strd libxsmm_ssbimmdispatch_reducebatch_strd(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.ssbimrs;
}


LIBXSMM_API libxsmm_usbimmfunction_reducebatch_strd libxsmm_usbimmdispatch_reducebatch_strd(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.usbimrs;
}


LIBXSMM_API libxsmm_subimmfunction_reducebatch_strd libxsmm_subimmdispatch_reducebatch_strd(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.subimrs;
}


LIBXSMM_API libxsmm_uubimmfunction_reducebatch_strd libxsmm_uubimmdispatch_reducebatch_strd(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_AB_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.uubimrs;
}


LIBXSMM_API libxsmm_sububmmfunction_reducebatch_strd libxsmm_sububmmdispatch_reducebatch_strd(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_C_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE,
    libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.sububmrs;
}


LIBXSMM_API libxsmm_dmmfunction_reducebatch_strd libxsmm_dmmdispatch_reducebatch_strd_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_dgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.dmrs;
}


LIBXSMM_API libxsmm_smmfunction_reducebatch_strd libxsmm_smmdispatch_reducebatch_strd_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? LIBXSMM_FLAGS : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_sgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.smrs;
}


LIBXSMM_API libxsmm_bsmmfunction_reducebatch_strd libxsmm_bsmmdispatch_reducebatch_strd_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.bsmrs;
}


LIBXSMM_API libxsmm_bmmfunction_reducebatch_strd libxsmm_bmmdispatch_reducebatch_strd_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.bmrs;
}


LIBXSMM_API libxsmm_wimmfunction_reducebatch_strd libxsmm_wimmdispatch_reducebatch_strd_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_wigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.wimrs;
}


LIBXSMM_API libxsmm_ssbimmfunction_reducebatch_strd libxsmm_ssbimmdispatch_reducebatch_strd_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.ssbimrs;
}


LIBXSMM_API libxsmm_usbimmfunction_reducebatch_strd libxsmm_usbimmdispatch_reducebatch_strd_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.usbimrs;
}


LIBXSMM_API libxsmm_subimmfunction_reducebatch_strd libxsmm_subimmdispatch_reducebatch_strd_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.subimrs;
}


LIBXSMM_API libxsmm_uubimmfunction_reducebatch_strd libxsmm_uubimmdispatch_reducebatch_strd_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bigemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_AB_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.uubimrs;
}


LIBXSMM_API libxsmm_sububmmfunction_reducebatch_strd libxsmm_sububmmdispatch_reducebatch_strd_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bbgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_B_UNSIGNED | LIBXSMM_GEMM_FLAG_C_UNSIGNED | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE,
    libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  result = libxsmm_xmmdispatch(desc);
  return result.sububmrs;
}


/* GEMMs fused with eltwise kernels */
LIBXSMM_API libxsmm_bmmfunction_reducebatch_strd_meltwfused libxsmm_bmmdispatch_reducebatch_strd_meltwfused(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxsmm_meltw_operation meltw_op, libxsmm_datatype meltw_dt, libxsmm_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  desc->meltw_datatype_aux = (unsigned char)meltw_dt;
  desc->meltw_flags = (unsigned short)meltw_flags;
  desc->meltw_operation = (unsigned char)meltw_op;
  desc->meltw_param = (unsigned char)meltw_param;
  desc->meltw_ldx = (unsigned int) meltw_ldx;
  desc->meltw_ldy = (unsigned int) meltw_ldy;
  desc->meltw_ldz = (unsigned int) meltw_ldz;
  result = libxsmm_xmmdispatch(desc);
  return result.bmrs_meltwfused;
}


LIBXSMM_API libxsmm_bmmfunction_reducebatch_strd_meltwfused libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxsmm_meltw_operation meltw_op, libxsmm_datatype meltw_dt, libxsmm_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  desc->meltw_datatype_aux = (unsigned char)meltw_dt;
  desc->meltw_flags = (unsigned short)meltw_flags;
  desc->meltw_operation = (unsigned char)meltw_op;
  desc->meltw_param = (unsigned char)meltw_param;
  desc->meltw_ldx = (unsigned int) meltw_ldx;
  desc->meltw_ldy = (unsigned int) meltw_ldy;
  desc->meltw_ldz = (unsigned int) meltw_ldz;
  result = libxsmm_xmmdispatch(desc);
  return result.bmrs_meltwfused;
}


LIBXSMM_API libxsmm_bsmmfunction_reducebatch_strd_meltwfused libxsmm_bsmmdispatch_reducebatch_strd_meltwfused(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxsmm_meltw_operation meltw_op, libxsmm_datatype meltw_dt, libxsmm_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  desc->meltw_datatype_aux = (unsigned char)meltw_dt;
  desc->meltw_flags = (unsigned short)meltw_flags;
  desc->meltw_operation = (unsigned char)meltw_op;
  desc->meltw_param = (unsigned char)meltw_param;
  desc->meltw_ldx = (unsigned int) meltw_ldx;
  desc->meltw_ldy = (unsigned int) meltw_ldy;
  desc->meltw_ldz = (unsigned int) meltw_ldz;
  result = libxsmm_xmmdispatch(desc);
  return result.bsmrs_meltwfused;
}


LIBXSMM_API libxsmm_bsmmfunction_reducebatch_strd_meltwfused libxsmm_bsmmdispatch_reducebatch_strd_meltwfused_unroll(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint stride_a, libxsmm_blasint stride_b, libxsmm_blasint unroll_hint,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc, const float* alpha, const float* beta, const int* flags, const int* prefetch,
  libxsmm_meltw_operation meltw_op, libxsmm_datatype meltw_dt, libxsmm_meltw_flags meltw_flags, unsigned char meltw_param, unsigned int meltw_ldx, unsigned int meltw_ldy, unsigned int meltw_ldz)
{
  const int gemm_flags = (NULL == flags ? (LIBXSMM_FLAGS | LIBXSMM_GEMM_FLAG_VNNI_A) : *flags);
  libxsmm_descriptor_blob blob;
  /*const*/ libxsmm_gemm_descriptor *const desc = libxsmm_bsgemm_descriptor_init(&blob, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    NULL != ldc ? *ldc : m, NULL != alpha ? *alpha : LIBXSMM_ALPHA, NULL != beta ? *beta : LIBXSMM_BETA,
    gemm_flags | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE, libxsmm_get_gemm_xprefetch(prefetch));
  /*const*/ libxsmm_xmmfunction result;
  desc->c1 = (unsigned long long)stride_a;
  desc->c2 = (unsigned long long)stride_b;
  desc->c3 = (unsigned char)(((unroll_hint < 255) && (unroll_hint > 0)) ? unroll_hint : 0);
  if ( (stride_a < 0) || (stride_b < 0) ) {
    return NULL;
  }
  desc->meltw_datatype_aux = (unsigned char)meltw_dt;
  desc->meltw_flags = (unsigned short)meltw_flags;
  desc->meltw_operation = (unsigned char)meltw_op;
  desc->meltw_param = (unsigned char)meltw_param;
  desc->meltw_ldx = (unsigned int) meltw_ldx;
  desc->meltw_ldy = (unsigned int) meltw_ldy;
  desc->meltw_ldz = (unsigned int) meltw_ldz;
  result = libxsmm_xmmdispatch(desc);
  return result.bsmrs_meltwfused;
}


LIBXSMM_API libxsmm_xmeltwfunction libxsmm_dispatch_meltw(const libxsmm_meltw_descriptor* descriptor)
{
  libxsmm_xmeltwfunction result;
  LIBXSMM_INIT /* verbosity */
#if !defined(LIBXSMM_UNPACKED) /* CCE/Classic */
  LIBXSMM_ASSERT((sizeof(*descriptor) + sizeof(libxsmm_descriptor_kind)) <= (LIBXSMM_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    libxsmm_descriptor wrap;
#if defined(LIBXSMM_UNPACKED) /* CCE/Classic */
    LIBXSMM_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
    LIBXSMM_ASSIGN127(&wrap.meltw.desc, descriptor);
    wrap.kind = LIBXSMM_DESCRIPTOR_BIG(LIBXSMM_KERNEL_KIND_MELTW);
    result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xmateltw;
  }
  else {
    result.xmeltw = NULL;
  }
  return result;
}


LIBXSMM_API libxsmm_meltwfunction_reduce_cols_idx libxsmm_dispatch_meltw_reduce_cols_idx(
  libxsmm_blasint m, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo,
  libxsmm_datatype in_type, libxsmm_datatype out_type, libxsmm_datatype idx_type)
{
  libxsmm_descriptor_blob blob;
  libxsmm_blasint idx_dtype_size = libxsmm_typesize(idx_type);
  const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init(&blob,
    in_type, out_type, m, idx_dtype_size, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    0, 0, LIBXSMM_MELTW_OPERATION_REDUCE_COLS_IDX);

  libxsmm_xmeltwfunction result = libxsmm_dispatch_meltw(desc);

  return result.meltw_reduce_cols_idx;
}


LIBXSMM_API libxsmm_meltwfunction_opreduce_vecs_idx libxsmm_dispatch_meltw_opreduce_vecs_idx(
  libxsmm_blasint m, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo,
  libxsmm_datatype in_type, libxsmm_datatype out_type, libxsmm_datatype idx_type, libxsmm_meltw_opreduce_vecs_flags flags)
{
  libxsmm_descriptor_blob blob;
  libxsmm_blasint idx_dtype_size = libxsmm_typesize(idx_type);
  unsigned char argidx_params = (unsigned char) (((flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_0) | (flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_1)) >> 16);
  const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init(&blob,
    in_type, out_type, m, idx_dtype_size, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo,
    (unsigned short)flags, argidx_params, LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX);

  libxsmm_xmeltwfunction result = libxsmm_dispatch_meltw(desc);

  return result.meltw_opreduce_vecs_idx;
}


LIBXSMM_API libxsmm_meltwfunction_unary libxsmm_dispatch_meltw_unary(
  libxsmm_blasint m, libxsmm_blasint n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldo,
  libxsmm_datatype in_type, libxsmm_datatype compute_type, libxsmm_datatype out_type, libxsmm_meltw_unary_flags flags, libxsmm_meltw_unary_type type)
{
  libxsmm_descriptor_blob blob;
  const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init2(&blob,
    in_type, compute_type, out_type, LIBXSMM_DATATYPE_UNSUPPORTED, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo, 0, 0,
    (unsigned short)flags, (unsigned char)type, LIBXSMM_MELTW_OPERATION_UNARY);

  libxsmm_xmeltwfunction result = libxsmm_dispatch_meltw(desc);

  return result.meltw_unary;
}


LIBXSMM_API libxsmm_meltwfunction_binary libxsmm_dispatch_meltw_binary(
  libxsmm_blasint m, libxsmm_blasint n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldi2, const libxsmm_blasint* ldo,
  libxsmm_datatype in_type, libxsmm_datatype compute_type, libxsmm_datatype out_type, libxsmm_meltw_binary_flags flags, libxsmm_meltw_binary_type type)
{
  libxsmm_descriptor_blob blob;
  const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init2(&blob,
    in_type, compute_type, out_type, LIBXSMM_DATATYPE_UNSUPPORTED, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo, (ldi2 == NULL) ? m : *ldi2, 0,
    (unsigned short)flags, (unsigned char)type, LIBXSMM_MELTW_OPERATION_BINARY);

  libxsmm_xmeltwfunction result = libxsmm_dispatch_meltw(desc);

  return result.meltw_binary;
}


LIBXSMM_API libxsmm_meltwfunction_ternary libxsmm_dispatch_meltw_ternary(
  libxsmm_blasint m, libxsmm_blasint n, const libxsmm_blasint* ldi, const libxsmm_blasint* ldi2, const libxsmm_blasint* ldi3, const libxsmm_blasint* ldo,
  libxsmm_datatype in_type, libxsmm_datatype compute_type, libxsmm_datatype out_type, libxsmm_meltw_ternary_flags flags, libxsmm_meltw_ternary_type type)
{
  libxsmm_descriptor_blob blob;
  const libxsmm_meltw_descriptor *const desc = libxsmm_meltw_descriptor_init2(&blob,
    in_type, compute_type, out_type, LIBXSMM_DATATYPE_UNSUPPORTED, m, n, (ldi == NULL) ? m : *ldi, (ldo == NULL) ? m : *ldo, (ldi2 == NULL) ? m : *ldi2, (ldi3 == NULL) ? m : *ldi3,
    (unsigned short)flags, (unsigned char)type, LIBXSMM_MELTW_OPERATION_TERNARY);

  libxsmm_xmeltwfunction result = libxsmm_dispatch_meltw(desc);

  return result.meltw_ternary;
}


LIBXSMM_API libxsmm_matrix_eqn_function libxsmm_dispatch_matrix_eqn_desc( const libxsmm_meqn_descriptor* descriptor ) {
  libxsmm_matrix_eqn_function result;
  LIBXSMM_INIT /* verbosity */
#if !defined(LIBXSMM_UNPACKED) /* CCE/Classic */
  LIBXSMM_ASSERT((sizeof(*descriptor) + sizeof(libxsmm_descriptor_kind)) <= (LIBXSMM_DESCRIPTOR_MAXSIZE));
#endif
  if (NULL != descriptor) {
    unsigned int hash;
    libxsmm_descriptor wrap;

    /* check if equation is ready for JIT */
    if ( libxsmm_matrix_eqn_is_ready_for_jit( descriptor->eqn_idx) == 0 ) {
#if defined(LIBXSMM_UNPACKED) /* CCE/Classic */
      LIBXSMM_MEMSET127(&wrap, 0, sizeof(*descriptor));
#endif
      LIBXSMM_ASSIGN127(&wrap.meqn.desc, descriptor);
      wrap.kind = LIBXSMM_DESCRIPTOR_BIG(LIBXSMM_KERNEL_KIND_MEQN);
      result = internal_find_code(&wrap, sizeof(*descriptor), 0/*user_size*/, &hash).xmateqn;
    } else {
      result = NULL;
    }
  }
  else {
    result = NULL;
  }
  return result;
}


LIBXSMM_API libxsmm_matrix_eqn_function libxsmm_dispatch_matrix_eqn(
  const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint* ldo,
  const libxsmm_datatype out_type, const unsigned int eqn_idx )
{
  libxsmm_descriptor_blob blob;
  const libxsmm_meqn_descriptor *const desc = libxsmm_meqn_descriptor_init(&blob,
    out_type, m, n, (ldo == NULL) ? m : *ldo, eqn_idx );

  return libxsmm_dispatch_matrix_eqn_desc( desc );
}


LIBXSMM_API libxsmm_xmmfunction libxsmm_create_packed_spxgemm_csr(const libxsmm_gemm_descriptor* descriptor, unsigned int packed_width,
  const unsigned int* row_ptr, const unsigned int* column_idx, const void* values)
{
  libxsmm_code_pointer result = { 0 };
  LIBXSMM_INIT
  if (NULL != descriptor && NULL != row_ptr && NULL != column_idx && NULL != values) {
    libxsmm_pspgemm_csr_descriptor pspgemm_csr;
    libxsmm_build_request request;
    libxsmm_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      pspgemm_csr.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      LIBXSMM_ASSIGN127(&desc, descriptor);
      desc.prefetch = (unsigned char)libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
      pspgemm_csr.gemm = &desc;
    }
    pspgemm_csr.row_ptr = row_ptr;
    pspgemm_csr.column_idx = column_idx;
    pspgemm_csr.values = values;
    pspgemm_csr.packed_width = packed_width;
    request.descriptor.pspgemm_csr = &pspgemm_csr;
    request.kind = LIBXSMM_BUILD_KIND_PSPGEMM_CSR;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXSMM_API libxsmm_xmmfunction libxsmm_create_packed_spxgemm_csc(const libxsmm_gemm_descriptor* descriptor, unsigned int packed_width,
  const unsigned int* column_ptr, const unsigned int* row_idx, const void* values)
{
  libxsmm_code_pointer result = { 0 };
  LIBXSMM_INIT
  if (NULL != descriptor && NULL != column_ptr && NULL != row_idx && NULL != values) {
    libxsmm_pspgemm_csc_descriptor pspgemm_csc;
    libxsmm_build_request request;
    libxsmm_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      pspgemm_csc.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      LIBXSMM_ASSIGN127(&desc, descriptor);
      desc.prefetch = (unsigned char)libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
      pspgemm_csc.gemm = &desc;
    }
    pspgemm_csc.column_ptr = column_ptr;
    pspgemm_csc.row_idx = row_idx;
    pspgemm_csc.values = values;
    pspgemm_csc.packed_width = packed_width;
    request.descriptor.pspgemm_csc = &pspgemm_csc;
    request.kind = LIBXSMM_BUILD_KIND_PSPGEMM_CSC;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXSMM_API libxsmm_xmmfunction libxsmm_create_packed_xgemm_ac_rm(const libxsmm_gemm_descriptor* descriptor, unsigned int packed_width)
{
  libxsmm_code_pointer result = { 0 };
  LIBXSMM_INIT
  if (NULL != descriptor) {
    libxsmm_pgemm_ac_rm_descriptor pgemmacrm;
    libxsmm_build_request request;
    libxsmm_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      pgemmacrm.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      LIBXSMM_ASSIGN127(&desc, descriptor);
      desc.prefetch = (unsigned char)libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
      pgemmacrm.gemm = &desc;
    }
    pgemmacrm.packed_width = packed_width;
    request.descriptor.pgemmacrm = &pgemmacrm;
    request.kind = LIBXSMM_BUILD_KIND_PGEMMRMAC;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXSMM_API libxsmm_xmmfunction libxsmm_create_packed_xgemm_bc_rm(const libxsmm_gemm_descriptor* descriptor, unsigned int packed_width)
{
  libxsmm_code_pointer result = { 0 };
  LIBXSMM_INIT
  if (NULL != descriptor) {
    libxsmm_pgemm_bc_rm_descriptor pgemmbcrm;
    libxsmm_build_request request;
    libxsmm_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      pgemmbcrm.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      LIBXSMM_ASSIGN127(&desc, descriptor);
      desc.prefetch = (unsigned char)libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
      pgemmbcrm.gemm = &desc;
    }
    pgemmbcrm.packed_width = packed_width;
    request.descriptor.pgemmbcrm = &pgemmbcrm;
    request.kind = LIBXSMM_BUILD_KIND_PGEMMRMBC;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm;
}


LIBXSMM_API libxsmm_dmmfunction libxsmm_create_dcsr_reg(const libxsmm_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const double* values)
{
  libxsmm_code_pointer result = { 0 };
  LIBXSMM_INIT
  if (NULL != descriptor && NULL != row_ptr && NULL != column_idx && NULL != values) {
    libxsmm_csr_reg_descriptor sreg;
    libxsmm_build_request request;
    libxsmm_gemm_descriptor desc;
    if (0 == (0x80 & descriptor->prefetch)) {
      sreg.gemm = descriptor;
    }
    else { /* "sign"-bit of byte-value is set */
      LIBXSMM_ASSIGN127(&desc, descriptor);
      desc.prefetch = (unsigned char)libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
      sreg.gemm = &desc;
    }
    sreg.row_ptr = row_ptr;
    sreg.column_idx = column_idx;
    sreg.values = values;
    request.descriptor.sreg = &sreg;
    request.kind = LIBXSMM_BUILD_KIND_SREG;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &result);
  }
  return result.xgemm.dmm;
}


LIBXSMM_API libxsmm_smmfunction libxsmm_create_scsr_reg(const libxsmm_gemm_descriptor* descriptor,
  const unsigned int* row_ptr, const unsigned int* column_idx, const float* values)
{
  libxsmm_code_pointer result = { 0 };
  LIBXSMM_INIT
  if (NULL != descriptor && NULL != row_ptr && NULL != column_idx && NULL != values) {
    libxsmm_csr_reg_descriptor sreg;
    libxsmm_build_request request;
    const unsigned int n = row_ptr[descriptor->m];
    double *const d_values = (double*)(0 != n ? malloc(n * sizeof(double)) : NULL);
    if (NULL != d_values) {
      libxsmm_gemm_descriptor desc;
      unsigned int i;
      /* we need to copy the values into a double precision buffer */
      for (i = 0; i < n; ++i) d_values[i] = (double)values[i];
      if (0 == (0x80 & descriptor->prefetch)) {
        sreg.gemm = descriptor;
      }
      else { /* "sign"-bit of byte-value is set */
        LIBXSMM_ASSIGN127(&desc, descriptor);
        desc.prefetch = (unsigned char)libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
        sreg.gemm = &desc;
      }
      sreg.row_ptr = row_ptr;
      sreg.column_idx = column_idx;
      sreg.values = d_values;
      request.descriptor.sreg = &sreg;
      request.kind = LIBXSMM_BUILD_KIND_SREG;
      libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &result);
      free(d_values);
    }
  }
  return result.xgemm.smm;
}


LIBXSMM_API void libxsmm_release_kernel(const void* kernel)
{
  if (NULL != kernel) {
    static int error_once = 0;
    libxsmm_kernel_xinfo* extra = NULL;
    void *const extra_address = &extra;
    LIBXSMM_INIT
    if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(
      kernel, NULL/*size*/, NULL/*flags*/, (void**)extra_address) && NULL != extra)
    {
      const unsigned int regindex = extra->registered;
      if ((LIBXSMM_CAPACITY_REGISTRY) <= regindex) {
        libxsmm_xfree(kernel, 0/*no check*/);
      }
      else { /* attempt to unregister kernel */
        libxsmm_kernel_info info;
#if !defined(LIBXSMM_ENABLE_DEREG)
        if (EXIT_SUCCESS == libxsmm_get_kernel_info(kernel, &info)
          && LIBXSMM_KERNEL_KIND_USER == info.kind)
#endif
        {
          LIBXSMM_ASSERT(LIBXSMM_KERNEL_UNREGISTERED > info.kind);
          /* coverity[check_return] */
          LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_ninit, 1, LIBXSMM_ATOMIC_RELAXED); /* invalidate code cache (TLS) */
          internal_registry[regindex].ptr = NULL;
#if !defined(NDEBUG)
          memset(internal_registry_keys + regindex, 0, sizeof(*internal_registry_keys));
#endif
          libxsmm_xfree(kernel, 0/*no check*/);
        }
#if !defined(LIBXSMM_ENABLE_DEREG)
        else if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM WARNING: attempt to unregister JIT-kernel!\n");
        }
#endif
      }
    }
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: failed to release kernel!\n");
    }
  }
}


#if defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_init)(void);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_init)(void)
{
  libxsmm_init();
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_finalize)(void);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_finalize)(void)
{
  libxsmm_finalize();
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_release_kernel)(const void** /*kernel*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_release_kernel)(const void** kernel)
{
#if !defined(NDEBUG)
  if (NULL != kernel)
#endif
  {
    libxsmm_release_kernel(*kernel);
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: invalid argument passed into libxsmm_release_kernel!\n");
    }
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xmmdispatch2)(intptr_t* /*fn*/, const int* /*iprec*/, const int* /*oprec*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*k*/,
  const libxsmm_blasint* /*lda*/, const libxsmm_blasint* /*ldb*/, const libxsmm_blasint* /*ldc*/,
  const void* /*alpha*/, const void* /*beta*/, const int* /*flags*/, const int* /*prefetch*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xmmdispatch2)(intptr_t* fn, const int* iprec, const int* oprec,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch)
{
#if !defined(NDEBUG)
  if (NULL != fn && NULL != m
    && (NULL == iprec || (0 <= *iprec && *iprec < LIBXSMM_DATATYPE_UNSUPPORTED))
    && (NULL == oprec || (0 <= *oprec && *oprec < LIBXSMM_DATATYPE_UNSUPPORTED)))
#endif
  {
    const int gemm_flags = (NULL != flags ? *flags : LIBXSMM_FLAGS);
    const libxsmm_gemm_descriptor* descriptor;
    libxsmm_gemm_prefetch_type gemm_prefetch;
    libxsmm_descriptor_blob blob;
    libxsmm_code_pointer result;
#if !defined(NDEBUG)
    const libxsmm_gemm_precision itype = (NULL != iprec ? ((libxsmm_gemm_precision)*iprec) : LIBXSMM_GEMM_PRECISION_F64);
    const libxsmm_gemm_precision otype = (NULL != oprec ? ((libxsmm_gemm_precision)*oprec) : itype);
    const libxsmm_blasint kk = *(NULL != k ? k : m), nn = (NULL != n ? *n : kk);
#else
    const libxsmm_gemm_precision itype = (libxsmm_gemm_precision)*iprec, otype = (libxsmm_gemm_precision)*oprec;
    const libxsmm_blasint kk = *k, nn = *n;
#endif
    LIBXSMM_PRAGMA_FORCEINLINE
    gemm_prefetch = libxsmm_get_gemm_xprefetch(prefetch);
    LIBXSMM_PRAGMA_FORCEINLINE
    descriptor = libxsmm_gemm_descriptor_init2(&blob, itype, otype, *m, nn, kk,
        NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? *m : kk),
        NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? kk : nn),
      *(NULL != ldc ? ldc : m), alpha, beta, gemm_flags, gemm_prefetch);
#if !defined(NDEBUG)
    if (NULL != descriptor)
#endif
    {
      LIBXSMM_PRAGMA_FORCEINLINE
      result.xgemm = libxsmm_xmmdispatch(descriptor);
      *fn = result.ival;
    }
#if !defined(NDEBUG)
    else { /* quiet */
      *fn = 0;
    }
#endif
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: invalid argument passed into libxsmm_xmmdispatch!\n");
    }
    if (NULL != fn) *fn = 0;
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xmmdispatch)(intptr_t* /*fn*/, const int* /*precision*/,
  const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*k*/,
  const libxsmm_blasint* /*lda*/, const libxsmm_blasint* /*ldb*/, const libxsmm_blasint* /*ldc*/,
  const void* /*alpha*/, const void* /*beta*/, const int* /*flags*/, const int* /*prefetch*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xmmdispatch)(intptr_t* fn, const int* precision,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch)
{
  LIBXSMM_FSYMBOL(libxsmm_xmmdispatch2)(fn, precision, precision, m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xmmcall_abc)(
  const libxsmm_xmmfunction* /*fn*/, const void* /*a*/, const void* /*b*/, void* /*c*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xmmcall_abc)(
  const libxsmm_xmmfunction* fn, const void* a, const void* b, void* c)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != fn && NULL != a && NULL != b && NULL != c)
#endif
  {
#if !defined(NDEBUG)
    if (NULL != fn->xmm)
#endif
    {
      fn->xmm(a, b, c);
    }
#if !defined(NDEBUG)
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: NULL-function passed into libxsmm_xmmcall_abc!\n");
    }
#endif
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xmmcall_abc specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xmmcall_prf)(
  const libxsmm_xmmfunction* /*fn*/, const void* /*a*/, const void* /*b*/, void* /*c*/,
  const void* /*pa*/, const void* /*pb*/, const void* /*pc*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xmmcall_prf)(
  const libxsmm_xmmfunction* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != fn && NULL != a && NULL != b && NULL != c)
#endif
  {
#if !defined(NDEBUG)
    if (NULL != fn->xmm)
#endif
    {
      fn->xmm(a, b, c, pa, pb, pc);
    }
#if !defined(NDEBUG)
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: NULL-function passed into libxsmm_xmmcall_prf!\n");
    }
#endif
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xmmcall_prf specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xmmcall)(
  const libxsmm_xmmfunction* /*fn*/, const void* /*a*/, const void* /*b*/, void* /*c*/,
  const void* /*pa*/, const void* /*pb*/, const void* /*pc*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xmmcall)(
  const libxsmm_xmmfunction* fn, const void* a, const void* b, void* c,
  const void* pa, const void* pb, const void* pc)
{
  LIBXSMM_FSYMBOL(libxsmm_xmmcall_prf)(fn, a, b, c, pa, pb, pc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xregister)(void** /*regval*/, const void* /*key*/, const int* /*keysize*/,
  const int* /*valsize*/, const void* /*valinit*/, int* /*keyhash*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xregister)(void** regval, const void* key, const int* keysize,
  const int* valsize, const void* valinit, int* keyhash)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != regval && NULL != key && NULL != keysize && NULL != valsize)
#endif
  {
    unsigned int hash = 0;
    *regval = libxsmm_xregister(key, *keysize, *valsize, valinit, &hash);
    if (NULL != keyhash) {
      *keyhash = (hash & 0x7FFFFFFF/*sign-bit*/);
    }
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xregister specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xdispatch)(void** /*regval*/, const void* /*key*/, const int* /*keysize*/, int* /*keyhash*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xdispatch)(void** regval, const void* key, const int* keysize, int* keyhash)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != regval && NULL != key && NULL != keysize)
#endif
  {
    unsigned int hash = 0;
    *regval = libxsmm_xdispatch(key, *keysize, &hash);
    if (NULL != keyhash) {
      *keyhash = (hash & 0x7FFFFFFF/*sign-bit*/);
    }
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xdispatch specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xrelease)(const void* /*key*/, const int* /*keysize*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xrelease)(const void* key, const int* keysize)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != key && NULL != keysize)
#endif
  {
    libxsmm_xrelease(key, *keysize);
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xrelease specified!\n");
  }
#endif
}

#endif /*defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/

