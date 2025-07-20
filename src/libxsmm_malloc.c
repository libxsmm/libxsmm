/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_trace.h"
#include "libxsmm_main.h"
#include "libxsmm_hash.h"

#if (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD)))
# include <features.h>
# include <malloc.h>
#endif
#if !defined(LIBXSMM_MALLOC_GLIBC)
# if defined(__GLIBC__)
#   define LIBXSMM_MALLOC_GLIBC __GLIBC__
# else
#   define LIBXSMM_MALLOC_GLIBC 6
# endif
#endif
#if defined(_WIN32)
# include <malloc.h>
# include <intrin.h>
#else
# include <sys/mman.h>
# if defined(__linux__)
#   include <linux/mman.h>
#   include <sys/syscall.h>
# endif
# if defined(MAP_POPULATE)
#   include <sys/utsname.h>
# endif
# include <sys/types.h>
# include <sys/stat.h>
# include <errno.h>
# if defined(__MAP_ANONYMOUS)
#   define LIBXSMM_MAP_ANONYMOUS __MAP_ANONYMOUS
# elif defined(MAP_ANONYMOUS)
#   define LIBXSMM_MAP_ANONYMOUS MAP_ANONYMOUS
# elif defined(MAP_ANON)
#   define LIBXSMM_MAP_ANONYMOUS MAP_ANON
# else
#  define LIBXSMM_MAP_ANONYMOUS 0x20
# endif
# if defined(MAP_SHARED)
#   define LIBXSMM_MAP_SHARED MAP_SHARED
# else
#   define LIBXSMM_MAP_SHARED 0
# endif
# if defined(MAP_JIT)
#   define LIBXSMM_MAP_JIT MAP_JIT
# else
#   define LIBXSMM_MAP_JIT 0
# endif
LIBXSMM_EXTERN int ftruncate(int, off_t) LIBXSMM_NOTHROW;
#endif
#if !defined(LIBXSMM_MALLOC_FINAL)
# define LIBXSMM_MALLOC_FINAL 3
#endif
#if defined(LIBXSMM_VTUNE)
# if (2 <= LIBXSMM_VTUNE) /* no header file required */
#   if !defined(LIBXSMM_VTUNE_JITVERSION)
#     define LIBXSMM_VTUNE_JITVERSION LIBXSMM_VTUNE
#   endif
#   define LIBXSMM_VTUNE_JIT_DESC_TYPE iJIT_Method_Load_V2
#   define LIBXSMM_VTUNE_JIT_LOAD 21
#   define LIBXSMM_VTUNE_JIT_UNLOAD 14
#   define iJIT_SAMPLING_ON 0x0001
LIBXSMM_EXTERN unsigned int iJIT_GetNewMethodID(void);
LIBXSMM_EXTERN /*iJIT_IsProfilingActiveFlags*/int iJIT_IsProfilingActive(void);
LIBXSMM_EXTERN int iJIT_NotifyEvent(/*iJIT_JVM_EVENT*/int event_type, void *EventSpecificData);
LIBXSMM_EXTERN_C typedef struct LineNumberInfo {
  unsigned int Offset;
  unsigned int LineNumber;
} LineNumberInfo;
LIBXSMM_EXTERN_C typedef struct iJIT_Method_Load_V2 {
  unsigned int method_id;
  char* method_name;
  void* method_load_address;
  unsigned int method_size;
  unsigned int line_number_size;
  LineNumberInfo* line_number_table;
  char* class_file_name;
  char* source_file_name;
  char* module_name;
} iJIT_Method_Load_V2;
# else /* more safe due to header dependency */
#   include <jitprofiling.h>
#   if !defined(LIBXSMM_VTUNE_JITVERSION)
#     define LIBXSMM_VTUNE_JITVERSION 2
#   endif
#   if (2 <= LIBXSMM_VTUNE_JITVERSION)
#     define LIBXSMM_VTUNE_JIT_DESC_TYPE iJIT_Method_Load_V2
#     define LIBXSMM_VTUNE_JIT_LOAD iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED_V2
#   else
#     define LIBXSMM_VTUNE_JIT_DESC_TYPE iJIT_Method_Load
#     define LIBXSMM_VTUNE_JIT_LOAD iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED
#   endif
#   define LIBXSMM_VTUNE_JIT_UNLOAD iJVM_EVENT_TYPE_METHOD_UNLOAD_START
# endif
# if !defined(LIBXSMM_MALLOC_FALLBACK)
#   define LIBXSMM_MALLOC_FALLBACK LIBXSMM_MALLOC_FINAL
# endif
#else /* VTune JIT-API not enabled */
# if !defined(LIBXSMM_MALLOC_FALLBACK)
#   define LIBXSMM_MALLOC_FALLBACK 0
# endif
#endif /*defined(LIBXSMM_VTUNE)*/
#if !defined(LIBXSMM_MALLOC_XMAP_TEMPLATE)
# define LIBXSMM_MALLOC_XMAP_TEMPLATE ".libxsmm_jit." LIBXSMM_MKTEMP_PATTERN
#endif
#if defined(LIBXSMM_PERF)
# include "libxsmm_perf.h"
#endif

#if !defined(LIBXSMM_MALLOC_ALIGNMAX)
# define LIBXSMM_MALLOC_ALIGNMAX (2 << 20) /* 2 MB */
#endif
#if !defined(LIBXSMM_MALLOC_ALIGNFCT)
# define LIBXSMM_MALLOC_ALIGNFCT 16
#endif
#if !defined(LIBXSMM_MALLOC_SEED)
# define LIBXSMM_MALLOC_SEED 1051981
#endif
#if !defined(LIBXSMM_MALLOC_NLOCKS)
# define LIBXSMM_MALLOC_NLOCKS 16
#endif

#if !defined(LIBXSMM_MALLOC_HOOK_KMP) && 0
# define LIBXSMM_MALLOC_HOOK_KMP
#endif
#if !defined(LIBXSMM_MALLOC_HOOK_QKMALLOC) && 0
# define LIBXSMM_MALLOC_HOOK_QKMALLOC
#endif
#if !defined(LIBXSMM_MALLOC_HOOK_IMALLOC) && 1
# define LIBXSMM_MALLOC_HOOK_IMALLOC
#endif
#if !defined(LIBXSMM_MALLOC_HOOK_CHECK) && 0
# define LIBXSMM_MALLOC_HOOK_CHECK 1
#endif

#if !defined(LIBXSMM_MALLOC_CRC_LIGHT) && !defined(_DEBUG) && 1
# define LIBXSMM_MALLOC_CRC_LIGHT
#endif
#if !defined(LIBXSMM_MALLOC_CRC_OFF)
# if defined(NDEBUG) && !defined(LIBXSMM_MALLOC_HOOK)
#   define LIBXSMM_MALLOC_CRC_OFF
# elif !defined(LIBXSMM_BUILD)
#   define LIBXSMM_MALLOC_CRC_OFF
# endif
#endif

#if !defined(LIBXSMM_MALLOC_SCRATCH_LIMIT)
# define LIBXSMM_MALLOC_SCRATCH_LIMIT 0xFFFFFFFF /* ~4 GB */
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_PADDING)
# define LIBXSMM_MALLOC_SCRATCH_PADDING LIBXSMM_CACHELINE
#endif
/* pointers are checked first if they belong to scratch */
#if !defined(LIBXSMM_MALLOC_SCRATCH_DELETE_FIRST) && 1
# define LIBXSMM_MALLOC_SCRATCH_DELETE_FIRST
#endif
/* can clobber memory if allocations are not exactly scoped */
#if !defined(LIBXSMM_MALLOC_SCRATCH_TRIM_HEAD) && 0
# define LIBXSMM_MALLOC_SCRATCH_TRIM_HEAD
#endif
#if !defined(LIBXSMM_MALLOC_SCRATCH_JOIN) && 1
# define LIBXSMM_MALLOC_SCRATCH_JOIN
#endif
#if !defined(LIBXSMM_MALLOC_HUGE_PAGES) && 1
# define LIBXSMM_MALLOC_HUGE_PAGES
#endif
#if !defined(LIBXSMM_MALLOC_LOCK_PAGES) && 1
/* 0: on-map, 1: mlock, 2: mlock2/on-fault */
# define LIBXSMM_MALLOC_LOCK_PAGES 1
#endif
#if !defined(LIBXSMM_MALLOC_LOCK_ALL) && \
     defined(LIBXSMM_MALLOC_MOD) && 0
# define LIBXSMM_MALLOC_LOCK_ALL
#endif
/* record real allocation size */
#if !defined(LIBXSMM_MALLOC_INFO_ALLOCSIZE) && 0
# define LIBXSMM_MALLOC_INFO_ALLOCSIZE
#endif
/* protected against double-delete (if possible) */
#if !defined(LIBXSMM_MALLOC_DELETE_SAFE) && 0
# define LIBXSMM_MALLOC_DELETE_SAFE
#elif !defined(NDEBUG)
# define LIBXSMM_MALLOC_DELETE_SAFE
#endif

#define INTERNAL_MEMALIGN_REAL(RESULT, ALIGNMENT, SIZE) do { \
  const size_t internal_memalign_real_alignment_ = INTERNAL_MALLOC_AUTOALIGN(SIZE, ALIGNMENT); \
  (RESULT) = (0 != internal_memalign_real_alignment_ \
    ? __real_memalign(internal_memalign_real_alignment_, SIZE) \
    : __real_malloc(SIZE)); \
} while(0)
#define INTERNAL_REALLOC_REAL(RESULT, PTR, SIZE) (RESULT) = __real_realloc(PTR, SIZE)
#define INTERNAL_FREE_REAL(PTR) __real_free(PTR)

#if defined(LIBXSMM_MALLOC_LOCK_ALL) && defined(LIBXSMM_MALLOC_LOCK_PAGES) && 0 != (LIBXSMM_MALLOC_LOCK_PAGES)
# if 1 == (LIBXSMM_MALLOC_LOCK_PAGES) || !defined(MLOCK_ONFAULT) || !defined(SYS_mlock2)
#   define INTERNAL_MALLOC_LOCK_PAGES(BUFFER, SIZE) if ((LIBXSMM_MALLOC_ALIGNFCT * LIBXSMM_MALLOC_ALIGNMAX) <= (SIZE)) \
      mlock(BUFFER, SIZE)
# else
#   define INTERNAL_MALLOC_LOCK_PAGES(BUFFER, SIZE) if ((LIBXSMM_MALLOC_ALIGNFCT * LIBXSMM_MALLOC_ALIGNMAX) <= (SIZE)) \
      syscall(SYS_mlock2, BUFFER, SIZE, MLOCK_ONFAULT)
# endif
#else
# define INTERNAL_MALLOC_LOCK_PAGES(BUFFER, SIZE)
#endif

#if defined(LIBXSMM_MALLOC_MOD)
# define INTERNAL_MALLOC_AUTOALIGN(SIZE, ALIGNMENT) libxsmm_alignment(SIZE, ALIGNMENT)
#else
# define INTERNAL_MALLOC_AUTOALIGN(SIZE, ALIGNMENT) (ALIGNMENT)
#endif

#if defined(LIBXSMM_MALLOC_HOOK) && defined(LIBXSMM_MALLOC) && (0 != LIBXSMM_MALLOC)
# define INTERNAL_MEMALIGN_HOOK(RESULT, FLAGS, ALIGNMENT, SIZE, CALLER) do { \
    const int internal_memalign_hook_recursive_ = LIBXSMM_ATOMIC_ADD_FETCH( \
      &internal_malloc_recursive, 1, LIBXSMM_ATOMIC_SEQ_CST); \
    if ( 1 < internal_memalign_hook_recursive_ /* protect against recursion */ \
      || 0 == (internal_malloc_kind & 1) || 0 >= internal_malloc_kind \
      || (internal_malloc_limit[0] > (SIZE)) \
      || (internal_malloc_limit[1] < (SIZE) && 0 != internal_malloc_limit[1])) \
    { \
      INTERNAL_MEMALIGN_REAL(RESULT, ALIGNMENT, SIZE); \
    } \
    else { /* redirect */ \
      LIBXSMM_INIT \
      if (NULL == (CALLER)) { /* libxsmm_trace_caller_id may allocate memory */ \
        internal_scratch_malloc(&(RESULT), SIZE, ALIGNMENT, FLAGS, \
          libxsmm_trace_caller_id(0/*level*/)); \
      } \
      else { \
        internal_scratch_malloc(&(RESULT), SIZE, ALIGNMENT, FLAGS, CALLER); \
      } \
    } \
    LIBXSMM_ATOMIC_SUB_FETCH(&internal_malloc_recursive, 1, LIBXSMM_ATOMIC_SEQ_CST); \
  } while(0)
# define INTERNAL_REALLOC_HOOK(RESULT, FLAGS, PTR, SIZE, CALLER) \
  if (0 == (internal_malloc_kind & 1) || 0 >= internal_malloc_kind \
    /*|| (0 != LIBXSMM_ATOMIC_LOAD(&internal_malloc_recursive, LIBXSMM_ATOMIC_SEQ_CST))*/ \
    || (internal_malloc_limit[0] > (SIZE)) \
    || (internal_malloc_limit[1] < (SIZE) && 0 != internal_malloc_limit[1])) \
  { \
    INTERNAL_REALLOC_REAL(RESULT, PTR, SIZE); \
  } \
  else { \
    const int nzeros = LIBXSMM_INTRINSICS_BITSCANFWD64((uintptr_t)(PTR)), alignment = 1 << nzeros; \
    LIBXSMM_ASSERT(0 == ((uintptr_t)(PTR) & ~(0xFFFFFFFFFFFFFFFF << nzeros))); \
    if (NULL == (CALLER)) { /* libxsmm_trace_caller_id may allocate memory */ \
      internal_scratch_malloc(&(PTR), SIZE, (size_t)alignment, FLAGS, \
        libxsmm_trace_caller_id(0/*level*/)); \
    } \
    else { \
      internal_scratch_malloc(&(PTR), SIZE, (size_t)alignment, FLAGS, CALLER); \
    } \
    (RESULT) = (PTR); \
  }
# define INTERNAL_FREE_HOOK(PTR, CALLER) do { \
    LIBXSMM_UNUSED(CALLER); \
    if (0 == (internal_malloc_kind & 1) || 0 >= internal_malloc_kind \
      /*|| (0 != LIBXSMM_ATOMIC_LOAD(&internal_malloc_recursive, LIBXSMM_ATOMIC_SEQ_CST))*/ \
    ) { \
      INTERNAL_FREE_REAL(PTR); \
    } \
    else { /* recognize pointers not issued by LIBXSMM */ \
      libxsmm_free(PTR); \
    } \
  } while(0)
#elif defined(LIBXSMM_MALLOC_MOD)
# define INTERNAL_MEMALIGN_HOOK(RESULT, FLAGS, ALIGNMENT, SIZE, CALLER) do { \
    LIBXSMM_UNUSED(FLAGS); LIBXSMM_UNUSED(CALLER); \
    INTERNAL_MEMALIGN_REAL(RESULT, ALIGNMENT, SIZE); \
    INTERNAL_MALLOC_LOCK_PAGES(RESULT, SIZE); \
  } while(0)
# define INTERNAL_REALLOC_HOOK(RESULT, FLAGS, PTR, SIZE, CALLER) do { \
    LIBXSMM_UNUSED(FLAGS); LIBXSMM_UNUSED(CALLER); \
    INTERNAL_REALLOC_REAL(RESULT, PTR, SIZE); \
    INTERNAL_MALLOC_LOCK_PAGES(RESULT, SIZE); \
  } while(0)
# define INTERNAL_FREE_HOOK(PTR, CALLER) do { \
    LIBXSMM_UNUSED(CALLER); \
    INTERNAL_FREE_REAL(PTR); \
  } while(0)
#endif

#if !defined(WIN32)
# if defined(MAP_32BIT)
#   define INTERNAL_XMALLOC_MAP32(ENV, MAPSTATE, MFLAGS, SIZE, BUFFER, REPTR) \
    if (MAP_FAILED == (BUFFER) && 0 != (MAP_32BIT & (MFLAGS))) do { \
      (BUFFER) = internal_xmalloc_xmap(ENV, SIZE, (MFLAGS) & ~MAP_32BIT, REPTR); \
      if (MAP_FAILED != (BUFFER)) (MAPSTATE) = 0; \
    } while(0)
# else
#   define INTERNAL_XMALLOC_MAP32(ENV, MAPSTATE, MFLAGS, SIZE, BUFFER, REPTR)
# endif

# define INTERNAL_XMALLOC(I, ENTRYPOINT, ENVVAR, ENVDEF, MAPSTATE, MFLAGS, SIZE, BUFFER, REPTR) \
  if ((ENTRYPOINT) <= (I) && (MAP_FAILED == (BUFFER) || NULL == (BUFFER))) do { \
    static const char* internal_xmalloc_env_ = NULL; \
    LIBXSMM_ASSERT(NULL != (ENVVAR) && '\0' != *(ENVVAR)); \
    if (NULL == internal_xmalloc_env_) { \
      internal_xmalloc_env_ = getenv(ENVVAR); \
      if (NULL == internal_xmalloc_env_) internal_xmalloc_env_ = ENVDEF; \
    } \
    (BUFFER) = internal_xmalloc_xmap(internal_xmalloc_env_, SIZE, MFLAGS, REPTR); \
    INTERNAL_XMALLOC_MAP32(internal_xmalloc_env_, MAPSTATE, MFLAGS, SIZE, BUFFER, REPTR); \
    if (MAP_FAILED != (BUFFER)) (ENTRYPOINT) = (I); \
  } while(0)

# define INTERNAL_XMALLOC_WATERMARK(NAME, WATERMARK, LIMIT, SIZE) do { \
  const size_t internal_xmalloc_watermark_ = (WATERMARK) + (SIZE) / 2; /* accept data-race */ \
  if (internal_xmalloc_watermark_ < (LIMIT)) { \
    static size_t internal_xmalloc_watermark_verbose_ = 0; \
    (LIMIT) = internal_xmalloc_watermark_; /* accept data-race */ \
    if (internal_xmalloc_watermark_verbose_ < internal_xmalloc_watermark_ && \
      (LIBXSMM_VERBOSITY_HIGH <= libxsmm_verbosity || 0 > libxsmm_verbosity)) \
    { /* muted */ \
      char internal_xmalloc_watermark_buffer_[32]; \
      /* coverity[check_return] */ \
      libxsmm_format_value(internal_xmalloc_watermark_buffer_, sizeof(internal_xmalloc_watermark_buffer_), \
        internal_xmalloc_watermark_, "KM", "B", 10); \
      fprintf(stderr, "LIBXSMM WARNING: " NAME " watermark reached at %s!\n", internal_xmalloc_watermark_buffer_); \
      internal_xmalloc_watermark_verbose_ = internal_xmalloc_watermark_; \
    } \
  } \
} while(0)

# define INTERNAL_XMALLOC_KIND(KIND, NAME, FLAG, FLAGS, MFLAGS, WATERMARK, LIMIT, INFO, SIZE, BUFFER) \
  if (0 != ((KIND) & (MFLAGS))) { \
    if (MAP_FAILED != (BUFFER)) { \
      LIBXSMM_ASSERT(NULL != (BUFFER)); \
      LIBXSMM_ATOMIC_ADD_FETCH(&(WATERMARK), SIZE, LIBXSMM_ATOMIC_RELAXED); \
      (FLAGS) |= (FLAG); \
    } \
    else { /* retry */ \
      (BUFFER) = mmap(NULL == (INFO) ? NULL : (INFO)->pointer, SIZE, PROT_READ | PROT_WRITE, \
        MAP_PRIVATE | LIBXSMM_MAP_ANONYMOUS | ((MFLAGS) & ~(KIND)), -1, 0/*offset*/); \
      if (MAP_FAILED != (BUFFER)) { /* successful retry */ \
        LIBXSMM_ASSERT(NULL != (BUFFER)); \
        INTERNAL_XMALLOC_WATERMARK(NAME, WATERMARK, LIMIT, SIZE); \
      } \
      (FLAGS) &= ~(FLAG); \
    } \
  } \
  else (FLAGS) &= ~(FLAG)
#endif


LIBXSMM_EXTERN_C typedef struct internal_malloc_info_type {
  libxsmm_free_function free;
  void *pointer, *reloc;
  const void* context;
#if defined(LIBXSMM_MALLOC_INFO_ALLOCSIZE)
  /* real/allocated size */
  size_t size_alloc;
#endif
  /* user-requested size */
  size_t size;
  int flags;
#if defined(LIBXSMM_VTUNE)
  unsigned int code_id;
#endif
#if !defined(LIBXSMM_MALLOC_CRC_OFF) /* hash *must* be the last entry */
  unsigned int hash;
#endif
} internal_malloc_info_type;

LIBXSMM_EXTERN_C typedef union internal_malloc_pool_type {
  char pad[LIBXSMM_MALLOC_SCRATCH_PADDING];
  struct {
    size_t minsize, counter, incsize;
    char *buffer, *head;
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
    const void* site;
# if (0 != LIBXSMM_SYNC)
    unsigned int tid;
# endif
#endif
  } instance;
} internal_malloc_pool_type;

/* Scratch pool, which supports up to MAX_NSCRATCH allocation sites. */
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
/* LIBXSMM_ALIGNED appears to contradict LIBXSMM_APIVAR, and causes multiple defined symbols (if below is seen in multiple translation units) */
LIBXSMM_APIVAR_DEFINE(char internal_malloc_pool_buffer[(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS)*sizeof(internal_malloc_pool_type)+(LIBXSMM_MALLOC_SCRATCH_PADDING)-1]);
#endif
/* Maximum total size of the scratch memory domain. */
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_scratch_limit);
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_scratch_nmallocs);
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_private_max);
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_private_cur);
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_public_max);
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_public_cur);
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_local_max);
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_local_cur);
LIBXSMM_APIVAR_DEFINE(int internal_malloc_recursive);
/** 0: regular, 1/odd: intercept/scratch, otherwise: all/scratch */
LIBXSMM_APIVAR_DEFINE(int internal_malloc_kind);
LIBXSMM_APIVAR_DEFINE(volatile int internal_pmallocs[LIBXSMM_MALLOC_NLOCKS]);
#if defined(LIBXSMM_MALLOC_HOOK) && defined(LIBXSMM_MALLOC) && (0 != LIBXSMM_MALLOC)
/* Interval of bytes that permit interception (internal_malloc_kind) */
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_limit[2]);
#endif
#if (0 != LIBXSMM_SYNC) && defined(LIBXSMM_MALLOC_SCRATCH_JOIN)
LIBXSMM_APIVAR_DEFINE(int internal_malloc_join);
#endif
#if !defined(_WIN32)
# if defined(MAP_HUGETLB) && defined(LIBXSMM_MALLOC_HUGE_PAGES)
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_hugetlb);
# endif
# if defined(MAP_LOCKED) && defined(LIBXSMM_MALLOC_LOCK_PAGES)
LIBXSMM_APIVAR_DEFINE(size_t internal_malloc_plocked);
# endif
#endif


LIBXSMM_API_INTERN size_t libxsmm_alignment(size_t size, size_t alignment)
{
  size_t result;
  if ((LIBXSMM_MALLOC_ALIGNFCT * LIBXSMM_MALLOC_ALIGNMAX) <= size) {
    result = libxsmm_lcm(0 == alignment ? (LIBXSMM_ALIGNMENT)
      : libxsmm_lcm(alignment, LIBXSMM_ALIGNMENT), LIBXSMM_MALLOC_ALIGNMAX);
  }
  else { /* small-size request */
    if ((LIBXSMM_MALLOC_ALIGNFCT * LIBXSMM_ALIGNMENT) <= size) {
      result = (0 == alignment ? (LIBXSMM_ALIGNMENT) : libxsmm_lcm(alignment, LIBXSMM_ALIGNMENT));
    }
    else if (0 != alignment) { /* custom alignment */
      result = libxsmm_lcm(alignment, sizeof(void*));
    }
    else { /* tiny-size request */
      result = sizeof(void*);
    }
  }
  return result;
}


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
LIBXSMM_API_INLINE
LIBXSMM_ATTRIBUTE_NO_SANITIZE(address)
internal_malloc_info_type* internal_malloc_info(const void* memory, int check)
{
  const char *const buffer = (const char*)memory;
  internal_malloc_info_type* result = (internal_malloc_info_type*)(NULL != memory
    ? (buffer - sizeof(internal_malloc_info_type)) : NULL);
#if defined(LIBXSMM_MALLOC_HOOK_CHECK)
  if ((LIBXSMM_MALLOC_HOOK_CHECK) < check) check = (LIBXSMM_MALLOC_HOOK_CHECK);
#endif
  if (0 != check && NULL != result) { /* check ownership */
#if !defined(_WIN32) /* mprotect: pass address rounded down to page-alignment */
    if (1 == check || 0 == mprotect((void*)(((uintptr_t)result) & ((uintptr_t)(-1 * LIBXSMM_PAGE_MINSIZE))),
      sizeof(internal_malloc_info_type), PROT_READ | PROT_WRITE) || ENOMEM != errno)
#endif
    {
      const int flags_rs = LIBXSMM_MALLOC_FLAG_REALLOC | LIBXSMM_MALLOC_FLAG_SCRATCH;
      const int flags_px = LIBXSMM_MALLOC_FLAG_X | LIBXSMM_MALLOC_FLAG_PRIVATE;
      const int flags_mx = LIBXSMM_MALLOC_FLAG_X | LIBXSMM_MALLOC_FLAG_MMAP;
      const char *const pointer = (const char*)result->pointer;
      union { libxsmm_free_fun fun; const void* ptr; } convert = { 0 };
      convert.fun = result->free.function;
      if (((flags_mx != (flags_mx & result->flags)) && NULL != result->reloc)
        || (0 == (LIBXSMM_MALLOC_FLAG_X & result->flags) ? 0 : (0 != (flags_rs & result->flags)))
        || (0 != (LIBXSMM_MALLOC_FLAG_X & result->flags) && NULL != result->context)
#if defined(LIBXSMM_VTUNE)
        || (0 == (LIBXSMM_MALLOC_FLAG_X & result->flags) && 0 != result->code_id)
#endif
        || (0 != (~LIBXSMM_MALLOC_FLAG_VALID & result->flags))
        || (0 == (LIBXSMM_MALLOC_FLAG_R & result->flags))
        || (pointer == convert.ptr || pointer == result->context || pointer >= buffer || NULL == pointer)
#if defined(LIBXSMM_MALLOC_INFO_ALLOCSIZE)
        || (result->size_alloc < result->size)
#endif
        || (LIBXSMM_MAX(LIBXSMM_MAX(internal_malloc_public_max, internal_malloc_local_max),
              internal_malloc_private_max) < result->size
            && 0 == (flags_px & result->flags)) || (0 == result->size)
        || (2 > libxsmm_ninit) /* before checksum calculation */
#if !defined(LIBXSMM_MALLOC_CRC_OFF) /* last check: checksum over info */
# if defined(LIBXSMM_MALLOC_CRC_LIGHT)
        || result->hash != LIBXSMM_CRCPTR(LIBXSMM_MALLOC_SEED, result)
# else
        || result->hash != libxsmm_crc32(LIBXSMM_MALLOC_SEED, result,
            (const char*)&result->hash - (const char*)result)
# endif
#endif
      ) { /* mismatch */
#if !defined(NDEBUG)
        if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
          static int error_once = 0;
          if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
            fprintf(stderr, "LIBXSMM ERROR: malloc/free mismatch!\n");
          }
        }
#endif
        result = NULL;
      }
    }
#if !defined(_WIN32)
    else { /* mismatch */
      result = NULL;
    }
#endif
  }
  return result;
}
#pragma GCC diagnostic pop


LIBXSMM_API_INLINE size_t internal_get_scratch_size(const internal_malloc_pool_type* exclude)
{
  size_t result = 0;
#if !defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) || (1 >= (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  LIBXSMM_UNUSED(exclude);
#else
  const internal_malloc_pool_type* pool = (const internal_malloc_pool_type*)LIBXSMM_UP2(
    (uintptr_t)internal_malloc_pool_buffer, LIBXSMM_MALLOC_SCRATCH_PADDING);
# if (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  const internal_malloc_pool_type *const end = pool + libxsmm_scratch_pools;
  LIBXSMM_ASSERT(libxsmm_scratch_pools <= LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS);
  for (; pool != end; ++pool)
# endif /*(1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))*/
  {
    if (0 != pool->instance.minsize) {
# if 1 /* memory info is not used */
      if (pool != exclude && (LIBXSMM_MALLOC_INTERNAL_CALLER) != pool->instance.site) {
        result += pool->instance.minsize;
      }
# else
      const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer, 0/*no check*/);
      if (NULL != info && pool != exclude && (LIBXSMM_MALLOC_INTERNAL_CALLER) != pool->instance.site) {
        result += info->size;
      }
# endif
    }
    else break; /* early exit */
  }
#endif /*defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))*/
  return result;
}


LIBXSMM_API_INLINE internal_malloc_pool_type* internal_scratch_malloc_pool(const void* memory)
{
  internal_malloc_pool_type* result = NULL;
  internal_malloc_pool_type* pool = (internal_malloc_pool_type*)LIBXSMM_UP2(
    (uintptr_t)internal_malloc_pool_buffer, LIBXSMM_MALLOC_SCRATCH_PADDING);
  const char *const buffer = (const char*)memory;
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  const unsigned int npools = libxsmm_scratch_pools;
#else
  const unsigned int npools = 1;
#endif
  internal_malloc_pool_type *const end = pool + npools;
  LIBXSMM_ASSERT(npools <= LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS);
  LIBXSMM_ASSERT(NULL != memory);
  for (; pool != end; ++pool) {
    if (0 != pool->instance.minsize) {
      if (0 != /*LIBXSMM_ATOMIC_LOAD(&*/pool->instance.counter/*, LIBXSMM_ATOMIC_SEQ_CST)*/
#if 1 /* should be implied by non-zero counter */
        && NULL != pool->instance.buffer
#endif
      ) {/* check if memory belongs to scratch domain or local domain */
#if 1
        const size_t size = pool->instance.minsize;
#else
        const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer, 0/*no check*/);
        const size_t size = info->size;
#endif
        if (pool->instance.buffer == buffer /* fast path */ ||
           (pool->instance.buffer < buffer && buffer < (pool->instance.buffer + size)))
        {
          result = pool;
          break;
        }
      }
    }
    else break; /* early exit */
  }
  return result;
}


LIBXSMM_API_INTERN int internal_xfree(const void* /*memory*/, internal_malloc_info_type* /*info*/);


LIBXSMM_API_INTERN void internal_scratch_free(const void* /*memory*/, internal_malloc_pool_type* /*pool*/);
LIBXSMM_API_INTERN void internal_scratch_free(const void* memory, internal_malloc_pool_type* pool)
{
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  const size_t counter = LIBXSMM_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXSMM_ATOMIC_SEQ_CST);
  char *const pool_buffer = pool->instance.buffer;
# if (!defined(NDEBUG) || defined(LIBXSMM_MALLOC_SCRATCH_TRIM_HEAD))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
  char *const buffer = (char*)memory; /* non-const */
#pragma GCC diagnostic pop
  LIBXSMM_ASSERT(pool_buffer <= buffer && buffer < pool_buffer + pool->instance.minsize);
# endif
  LIBXSMM_ASSERT(pool_buffer <= pool->instance.head);
  if (0 == counter) { /* reuse or reallocate scratch domain */
    internal_malloc_info_type *const info = internal_malloc_info(pool_buffer, 0/*no check*/);
    const size_t scale_size = (size_t)(1 != libxsmm_scratch_scale ? (libxsmm_scratch_scale * info->size) : info->size); /* hysteresis */
    const size_t size = pool->instance.minsize + pool->instance.incsize;
    LIBXSMM_ASSERT(0 == (LIBXSMM_MALLOC_FLAG_X & info->flags)); /* scratch memory is not executable */
    if (size <= scale_size) { /* reuse scratch domain */
      pool->instance.head = pool_buffer; /* reuse scratch domain */
    }
    else { /* release buffer */
# if !defined(NDEBUG)
      static int error_once = 0;
# endif
      pool->instance.buffer = pool->instance.head = NULL;
# if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
      pool->instance.site = NULL; /* clear affinity */
# endif
# if !defined(NDEBUG)
      if (EXIT_SUCCESS != internal_xfree(pool_buffer, info) /* invalidates info */
        && 0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: memory deallocation failed!\n");
      }
# else
      internal_xfree(pool_buffer, info); /* !libxsmm_free, invalidates info */
# endif
    }
  }
# if defined(LIBXSMM_MALLOC_SCRATCH_TRIM_HEAD) /* TODO: document linear/scoped allocator policy */
  else if (buffer < pool->instance.head) { /* reuse scratch domain */
    pool->instance.head = buffer;
  }
# else
  LIBXSMM_UNUSED(memory);
# endif
#else
  LIBXSMM_UNUSED(memory); LIBXSMM_UNUSED(pool);
#endif
}


LIBXSMM_API_INTERN void internal_scratch_malloc(void** /*memory*/, size_t /*size*/, size_t /*alignment*/, int /*flags*/, const void* /*caller*/);
LIBXSMM_API_INTERN void internal_scratch_malloc(void** memory, size_t size, size_t alignment, int flags, const void* caller)
{
  LIBXSMM_ASSERT(NULL != memory && 0 == (LIBXSMM_MALLOC_FLAG_X & flags));
  if (0 == (LIBXSMM_MALLOC_FLAG_REALLOC & flags) || NULL == *memory) {
    static int error_once = 0;
    size_t local_size = 0;
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
    if (0 < libxsmm_scratch_pools) {
      internal_malloc_pool_type *const pools = (internal_malloc_pool_type*)LIBXSMM_UP2(
        (uintptr_t)internal_malloc_pool_buffer, LIBXSMM_MALLOC_SCRATCH_PADDING);
      internal_malloc_pool_type *const end = pools + libxsmm_scratch_pools, *pool = pools;
      const size_t align_size = libxsmm_alignment(size, alignment), alloc_size = size + align_size - 1;
# if (0 != LIBXSMM_SYNC)
      const unsigned int tid = libxsmm_get_tid();
# endif
      unsigned int npools = 1;
# if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
      const void *const site = caller; /* no further attempt in case of NULL */
      internal_malloc_pool_type *pool0 = end;
      for (; pool != end; ++pool) { /* counter: memory info is not employed as pools are still manipulated */
        if (NULL != pool->instance.buffer) {
          if ((LIBXSMM_MALLOC_INTERNAL_CALLER) != pool->instance.site) ++npools; /* count number of occupied pools */
          if ( /* find matching pool and enter fast path (draw from pool-buffer) */
#   if (0 != LIBXSMM_SYNC) && !defined(LIBXSMM_MALLOC_SCRATCH_JOIN)
            (site == pool->instance.site && tid == pool->instance.tid))
#   elif (0 != LIBXSMM_SYNC)
            (site == pool->instance.site && (0 != internal_malloc_join || tid == pool->instance.tid)))
#   else
            (site == pool->instance.site))
#   endif
          {
            break;
          }
        }
        else {
          if (end == pool0) pool0 = pool; /* first available pool*/
          if (0 == pool->instance.minsize) { /* early exit */
            pool = pool0; break;
          }
        }
      }
# endif
      LIBXSMM_ASSERT(NULL != pool);
      if (end != pool && 0 <= internal_malloc_kind) {
        const size_t counter = LIBXSMM_ATOMIC_ADD_FETCH(&pool->instance.counter, (size_t)1, LIBXSMM_ATOMIC_SEQ_CST);
        if (NULL != pool->instance.buffer || 1 != counter) { /* attempt to (re-)use existing pool */
          const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer, 1/*check*/);
          const size_t pool_size = ((NULL != info && 0 != counter) ? info->size : 0);
          const size_t used_size = pool->instance.head - pool->instance.buffer;
          const size_t req_size = alloc_size + used_size;
          if (req_size <= pool_size) { /* fast path: draw from pool-buffer */
# if (0 != LIBXSMM_SYNC) && defined(LIBXSMM_MALLOC_SCRATCH_JOIN)
            void *const headaddr = &pool->instance.head;
            const uintptr_t headint = (0 == internal_malloc_join
              ? (uintptr_t)(pool->instance.head += alloc_size)
              : (LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_ADD_FETCH, LIBXSMM_BITS)(
                (uintptr_t*)headaddr, alloc_size, LIBXSMM_ATOMIC_SEQ_CST)));
            char *const head = (char*)headint;
# else
            char *const head = (char*)(pool->instance.head += alloc_size);
# endif
            *memory = LIBXSMM_ALIGN(head - alloc_size, align_size);
          }
          else { /* fallback to local memory allocation */
            const size_t incsize = req_size - LIBXSMM_MIN(pool_size, req_size);
            pool->instance.incsize = LIBXSMM_MAX(pool->instance.incsize, incsize);
# if (0 != LIBXSMM_SYNC) && defined(LIBXSMM_MALLOC_SCRATCH_JOIN)
            if (0 == internal_malloc_join) {
              --pool->instance.counter;
            }
            else {
              LIBXSMM_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXSMM_ATOMIC_SEQ_CST);
            }
# else
            --pool->instance.counter;
# endif
            if (
# if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
              (LIBXSMM_MALLOC_INTERNAL_CALLER) != pool->instance.site &&
# endif
              0 == (LIBXSMM_MALLOC_FLAG_PRIVATE & flags))
            {
              const size_t watermark = LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_ADD_FETCH, LIBXSMM_BITS)(
                &internal_malloc_local_cur, alloc_size, LIBXSMM_ATOMIC_RELAXED);
              if (internal_malloc_local_max < watermark) internal_malloc_local_max = watermark; /* accept data-race */
            }
            else {
              const size_t watermark = LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_ADD_FETCH, LIBXSMM_BITS)(
                &internal_malloc_private_cur, alloc_size, LIBXSMM_ATOMIC_RELAXED);
              if (internal_malloc_private_max < watermark) internal_malloc_private_max = watermark; /* accept data-race */
            }
            local_size = size;
          }
        }
        else { /* fresh pool */
          const size_t scratch_limit = libxsmm_get_scratch_limit();
          const size_t scratch_size = internal_get_scratch_size(pool); /* exclude current pool */
          const size_t limit_size = (1 < npools ? (scratch_limit - LIBXSMM_MIN(scratch_size, scratch_limit)) : LIBXSMM_SCRATCH_UNLIMITED);
          const size_t scale_size = (size_t)(1 != libxsmm_scratch_scale ? (libxsmm_scratch_scale * alloc_size) : alloc_size); /* hysteresis */
          const size_t incsize = (size_t)(libxsmm_scratch_scale * pool->instance.incsize);
          const size_t maxsize = LIBXSMM_MAX(scale_size, pool->instance.minsize) + incsize;
          const size_t limsize = LIBXSMM_MIN(maxsize, limit_size);
          const size_t minsize = limsize;
          assert(1 <= libxsmm_scratch_scale); /* !LIBXSMM_ASSERT */
          LIBXSMM_ASSERT(1 == counter);
          pool->instance.incsize = 0; /* reset */
          pool->instance.minsize = minsize;
# if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
          pool->instance.site = site;
#   if (0 != LIBXSMM_SYNC)
          pool->instance.tid = tid;
#   endif
# endif
          if (alloc_size <= minsize && /* allocate scratch pool */
            EXIT_SUCCESS == libxsmm_xmalloc(memory, minsize, 0/*auto-align*/,
              (flags | LIBXSMM_MALLOC_FLAG_SCRATCH) & ~LIBXSMM_MALLOC_FLAG_REALLOC,
              NULL/*extra*/, 0/*extra_size*/))
          {
            pool->instance.buffer = (char*)*memory;
            pool->instance.head = pool->instance.buffer + alloc_size;
            *memory = LIBXSMM_ALIGN((char*)*memory, align_size);
# if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
            if ((LIBXSMM_MALLOC_INTERNAL_CALLER) != pool->instance.site)
# endif
            {
              LIBXSMM_ATOMIC_ADD_FETCH(&internal_malloc_scratch_nmallocs, 1, LIBXSMM_ATOMIC_RELAXED);
            }
          }
          else { /* fallback to local allocation */
            LIBXSMM_ATOMIC_SUB_FETCH(&pool->instance.counter, 1, LIBXSMM_ATOMIC_SEQ_CST);
            if (0 != libxsmm_verbosity /* library code is expected to be mute */
              && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
            {
              if (alloc_size <= minsize) {
                fprintf(stderr, "LIBXSMM ERROR: failed to allocate scratch memory!\n");
              }
              else if ((LIBXSMM_MALLOC_INTERNAL_CALLER) != caller
                && (LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity))
              {
                fprintf(stderr, "LIBXSMM WARNING: scratch memory domain exhausted!\n");
              }
            }
            local_size = size;
          }
        }
      }
      else { /* fallback to local memory allocation */
        local_size = size;
      }
    }
    else { /* fallback to local memory allocation */
      local_size = size;
    }
    if (0 != local_size)
#else
    local_size = size;
#endif /*defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))*/
    { /* local memory allocation */
      if (EXIT_SUCCESS != libxsmm_xmalloc(memory, local_size, alignment,
          flags & ~(LIBXSMM_MALLOC_FLAG_SCRATCH | LIBXSMM_MALLOC_FLAG_REALLOC), NULL/*extra*/, 0/*extra_size*/)
        && /* library code is expected to be mute */0 != libxsmm_verbosity
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: scratch memory fallback failed!\n");
        assert(NULL == *memory); /* !LIBXSMM_ASSERT */
      }
      if ((LIBXSMM_MALLOC_INTERNAL_CALLER) != caller) {
        LIBXSMM_ATOMIC_ADD_FETCH(&internal_malloc_scratch_nmallocs, 1, LIBXSMM_ATOMIC_RELAXED);
      }
    }
  }
  else { /* reallocate memory */
    const void *const preserve = *memory;
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
    internal_malloc_pool_type *const pool = internal_scratch_malloc_pool(preserve);
    if (NULL != pool) {
      const internal_malloc_info_type *const info = internal_malloc_info(pool->instance.buffer, 0/*no check*/);
      void* buffer;
      LIBXSMM_ASSERT(pool->instance.buffer <= pool->instance.head && NULL != info);
      internal_scratch_malloc(&buffer, size, alignment,
        ~LIBXSMM_MALLOC_FLAG_REALLOC & (LIBXSMM_MALLOC_FLAG_SCRATCH | flags), caller);
      if (NULL != buffer) {
        memcpy(buffer, preserve, LIBXSMM_MIN(size, info->size)); /* TODO: memmove? */
        *memory = buffer;
      }
      internal_scratch_free(memory, pool);
    }
    else
#endif
    { /* non-pooled (potentially foreign pointer) */
#if !defined(NDEBUG)
      const int status =
#endif
      libxsmm_xmalloc(memory, size, alignment/* no need here to determine alignment of given buffer */,
        ~LIBXSMM_MALLOC_FLAG_SCRATCH & flags, NULL/*extra*/, 0/*extra_size*/);
      assert(EXIT_SUCCESS == status || NULL == *memory); /* !LIBXSMM_ASSERT */
    }
  }
}


#if defined(LIBXSMM_MALLOC_HOOK_DYNAMIC)
LIBXSMM_APIVAR_PRIVATE_DEF(libxsmm_malloc_fntype libxsmm_malloc_fn);

#if defined(LIBXSMM_MALLOC_HOOK_QKMALLOC)
LIBXSMM_API_INTERN void* internal_memalign_malloc(size_t /*alignment*/, size_t /*size*/);
LIBXSMM_API_INTERN void* internal_memalign_malloc(size_t alignment, size_t size)
{
  LIBXSMM_UNUSED(alignment);
  LIBXSMM_ASSERT(NULL != libxsmm_malloc_fn.malloc.dlsym);
  return libxsmm_malloc_fn.malloc.ptr(size);
}
#elif defined(LIBXSMM_MALLOC_HOOK_KMP)
LIBXSMM_API_INTERN void* internal_memalign_twiddle(size_t /*alignment*/, size_t /*size*/);
LIBXSMM_API_INTERN void* internal_memalign_twiddle(size_t alignment, size_t size)
{
  LIBXSMM_ASSERT(NULL != libxsmm_malloc_fn.alignmem.dlsym);
  return libxsmm_malloc_fn.alignmem.ptr(size, alignment);
}
#endif
#endif /*defined(LIBXSMM_MALLOC_HOOK_DYNAMIC)*/


#if (defined(LIBXSMM_MALLOC_HOOK) && defined(LIBXSMM_MALLOC) && (0 != LIBXSMM_MALLOC)) || defined(LIBXSMM_MALLOC_MOD)
LIBXSMM_API_INTERN void* internal_memalign_hook(size_t /*alignment*/, size_t /*size*/, const void* /*caller*/);
LIBXSMM_API_INTERN void* internal_memalign_hook(size_t alignment, size_t size, const void* caller)
{
  void* result;
# if defined(LIBXSMM_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_MMAP, alignment, size, caller);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_DEFAULT, alignment, size, caller);
# endif
  return result;
}

LIBXSMM_API void* __wrap_memalign(size_t /*alignment*/, size_t /*size*/);
LIBXSMM_API void* __wrap_memalign(size_t alignment, size_t size)
{
  void* result;
# if defined(LIBXSMM_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_MMAP, alignment, size, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_DEFAULT, alignment, size, NULL/*caller*/);
# endif
  return result;
}

LIBXSMM_API_INTERN void* internal_malloc_hook(size_t /*size*/, const void* /*caller*/);
LIBXSMM_API_INTERN void* internal_malloc_hook(size_t size, const void* caller)
{
  return internal_memalign_hook(0/*auto-alignment*/, size, caller);
}

LIBXSMM_API void* __wrap_malloc(size_t /*size*/);
LIBXSMM_API void* __wrap_malloc(size_t size)
{
  void* result;
# if defined(LIBXSMM_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_MMAP, 0/*auto-alignment*/, size, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_DEFAULT, 0/*auto-alignment*/, size, NULL/*caller*/);
# endif
  return result;
}

#if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
LIBXSMM_API void* __wrap_calloc(size_t /*num*/, size_t /*size*/);
LIBXSMM_API void* __wrap_calloc(size_t num, size_t size)
{
  void* result;
  const size_t nbytes = num * size;
# if defined(LIBXSMM_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_MMAP, 0/*auto-alignment*/, nbytes, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_DEFAULT, 0/*auto-alignment*/, nbytes, NULL/*caller*/);
# endif
  /* TODO: signal anonymous/zeroed pages */
  if (NULL != result) memset(result, 0, nbytes);
  return result;
}
#endif

#if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
LIBXSMM_API_INTERN void* internal_realloc_hook(void* /*ptr*/, size_t /*size*/, const void* /*caller*/);
LIBXSMM_API_INTERN void* internal_realloc_hook(void* ptr, size_t size, const void* caller)
{
  void* result;
# if defined(LIBXSMM_MALLOC_MMAP_HOOK)
  INTERNAL_REALLOC_HOOK(result, LIBXSMM_MALLOC_FLAG_REALLOC | LIBXSMM_MALLOC_FLAG_MMAP, ptr, size, caller);
# else
  INTERNAL_REALLOC_HOOK(result, LIBXSMM_MALLOC_FLAG_REALLOC | LIBXSMM_MALLOC_FLAG_DEFAULT, ptr, size, caller);
# endif
  return result;
}

LIBXSMM_API void* __wrap_realloc(void* /*ptr*/, size_t /*size*/);
LIBXSMM_API void* __wrap_realloc(void* ptr, size_t size)
{
  void* result;
# if defined(LIBXSMM_MALLOC_MMAP_HOOK)
  INTERNAL_REALLOC_HOOK(result, LIBXSMM_MALLOC_FLAG_REALLOC | LIBXSMM_MALLOC_FLAG_MMAP, ptr, size, NULL/*caller*/);
# else
  INTERNAL_REALLOC_HOOK(result, LIBXSMM_MALLOC_FLAG_REALLOC | LIBXSMM_MALLOC_FLAG_DEFAULT, ptr, size, NULL/*caller*/);
# endif
  return result;
}
#endif

LIBXSMM_API_INTERN void internal_free_hook(void* /*ptr*/, const void* /*caller*/);
LIBXSMM_API_INTERN void internal_free_hook(void* ptr, const void* caller)
{
  INTERNAL_FREE_HOOK(ptr, caller);
}

LIBXSMM_API void __wrap_free(void* /*ptr*/);
LIBXSMM_API void __wrap_free(void* ptr)
{
  INTERNAL_FREE_HOOK(ptr, NULL/*caller*/);
}
#endif

#if defined(LIBXSMM_MALLOC_HOOK_DYNAMIC) && ((defined(LIBXSMM_MALLOC) && (0 != LIBXSMM_MALLOC)) || defined(LIBXSMM_MALLOC_MOD))
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK LIBXSMM_ATTRIBUTE_MALLOC void* memalign(size_t /*alignment*/, size_t /*size*/) LIBXSMM_NOTHROW;
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK LIBXSMM_ATTRIBUTE_MALLOC void* memalign(size_t alignment, size_t size) LIBXSMM_NOEXCEPT
{
  void* result;
# if defined(LIBXSMM_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_MMAP, alignment, size, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_DEFAULT, alignment, size, NULL/*caller*/);
# endif
  return result;
}

LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK LIBXSMM_ATTRIBUTE_MALLOC void* malloc(size_t /*size*/) LIBXSMM_NOTHROW;
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK LIBXSMM_ATTRIBUTE_MALLOC void* malloc(size_t size) LIBXSMM_NOEXCEPT
{
  void* result;
# if defined(LIBXSMM_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_MMAP, 0/*auto-alignment*/, size, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_DEFAULT, 0/*auto-alignment*/, size, NULL/*caller*/);
# endif
  return result;
}

#if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK LIBXSMM_ATTRIBUTE_MALLOC void* calloc(size_t /*num*/, size_t /*size*/) LIBXSMM_NOTHROW;
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK LIBXSMM_ATTRIBUTE_MALLOC void* calloc(size_t num, size_t size) LIBXSMM_NOEXCEPT
{
  void* result;
  const size_t nbytes = num * size;
# if defined(LIBXSMM_MALLOC_MMAP_HOOK)
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_MMAP, 0/*auto-alignment*/, nbytes, NULL/*caller*/);
# else
  INTERNAL_MEMALIGN_HOOK(result, LIBXSMM_MALLOC_FLAG_DEFAULT, 0/*auto-alignment*/, nbytes, NULL/*caller*/);
# endif
  /* TODO: signal anonymous/zeroed pages */
  if (NULL != result) memset(result, 0, nbytes);
  return result;
}
#endif

#if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void* realloc(void* /*ptr*/, size_t /*size*/) LIBXSMM_NOTHROW;
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void* realloc(void* ptr, size_t size) LIBXSMM_NOEXCEPT
{
  void* result;
# if defined(LIBXSMM_MALLOC_MMAP_HOOK)
  INTERNAL_REALLOC_HOOK(result, LIBXSMM_MALLOC_FLAG_REALLOC | LIBXSMM_MALLOC_FLAG_MMAP, ptr, size, NULL/*caller*/);
# else
  INTERNAL_REALLOC_HOOK(result, LIBXSMM_MALLOC_FLAG_REALLOC | LIBXSMM_MALLOC_FLAG_DEFAULT, ptr, size, NULL/*caller*/);
# endif
  return result;
}
#endif

LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void free(void* /*ptr*/) LIBXSMM_NOTHROW;
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void free(void* ptr) LIBXSMM_NOEXCEPT
{
  INTERNAL_FREE_HOOK(ptr, NULL/*caller*/);
}
#endif


LIBXSMM_API_INTERN int internal_xfree(const void* memory, internal_malloc_info_type* info)
{
#if !defined(LIBXSMM_BUILD) || !defined(_WIN32)
  static int error_once = 0;
#endif
  int result = EXIT_SUCCESS;
  internal_malloc_info_type local /*= { 0 }*/;
  LIBXSMM_ASSIGN127(&local, info);
#if !defined(LIBXSMM_BUILD) /* sanity check */
  if (NULL != local.pointer || 0 == local.size)
#endif
  {
#if !defined(LIBXSMM_MALLOC_INFO_ALLOCSIZE) || !defined(NDEBUG)
    const size_t size = local.size + (size_t)(((const char*)memory) - ((const char*)local.pointer));
#endif
#if defined(LIBXSMM_MALLOC_INFO_ALLOCSIZE)
    const size_t size_alloc = local.size_alloc;
    assert(0 == local.size || (NULL != local.pointer && size <= size_alloc)); /* !LIBXSMM_ASSERT */
#else
    const size_t size_alloc = /*LIBXSMM_UP2(*/size/*, LIBXSMM_PAGE_MINSIZE)*/;
#endif
    assert(NULL != memory && NULL != info && sizeof(internal_malloc_info_type) < size_alloc); /* !LIBXSMM_ASSERT */
#if defined(LIBXSMM_MALLOC_INFO_ALLOCSIZE) && defined(NDEBUG)
    LIBXSMM_UNUSED(memory);
#endif
    if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & local.flags)) {
      if (NULL != local.free.function) {
#if defined(LIBXSMM_MALLOC_DELETE_SAFE)
        LIBXSMM_MEMZERO127(info);
#endif
        if (NULL == local.context) {
#if defined(LIBXSMM_MALLOC_HOOK)
          if (free == local.free.function) {
            __real_free(local.pointer);
          }
          else
#endif
          if (NULL != local.free.function) {
            local.free.function(local.pointer);
          }
        }
        else {
          LIBXSMM_ASSERT(NULL != local.free.ctx_form);
          local.free.ctx_form(local.pointer, local.context);
        }
      }
    }
    else {
#if defined(LIBXSMM_VTUNE)
      if (0 != (LIBXSMM_MALLOC_FLAG_X & local.flags) && 0 != local.code_id && iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
        iJIT_NotifyEvent(LIBXSMM_VTUNE_JIT_UNLOAD, &local.code_id);
      }
#endif
#if defined(_WIN32)
      result = (NULL == local.pointer || FALSE != VirtualFree(local.pointer, 0, MEM_RELEASE)) ? EXIT_SUCCESS : EXIT_FAILURE;
#else /* !_WIN32 */
      {
        if (0 != munmap(local.pointer, size_alloc)) {
          if (0 != libxsmm_verbosity /* library code is expected to be mute */
            && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXSMM ERROR: %s (attempted to unmap buffer %p+%" PRIuPTR ")!\n",
              strerror(errno), local.pointer, (uintptr_t)size_alloc);
          }
          result = EXIT_FAILURE;
        }
        if (0 != (LIBXSMM_MALLOC_FLAG_X & local.flags) && EXIT_SUCCESS == result
          && NULL != local.reloc && MAP_FAILED != local.reloc && local.pointer != local.reloc
          && 0 != munmap(local.reloc, size_alloc))
        {
          if (0 != libxsmm_verbosity /* library code is expected to be mute */
            && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXSMM ERROR: %s (attempted to unmap code %p+%" PRIuPTR ")!\n",
              strerror(errno), local.reloc, (uintptr_t)size_alloc);
          }
          result = EXIT_FAILURE;
        }
      }
#endif
    }
    if (0 == (LIBXSMM_MALLOC_FLAG_X & local.flags)) { /* update statistics */
#if !defined(_WIN32)
# if defined(MAP_HUGETLB) && defined(LIBXSMM_MALLOC_HUGE_PAGES)
      if (0 != (LIBXSMM_MALLOC_FLAG_PHUGE & local.flags)) { /* huge pages */
        LIBXSMM_ASSERT(0 != (LIBXSMM_MALLOC_FLAG_MMAP & local.flags));
        LIBXSMM_ATOMIC_SUB_FETCH(&internal_malloc_hugetlb, size_alloc, LIBXSMM_ATOMIC_RELAXED);
      }
# endif
# if defined(MAP_LOCKED) && defined(LIBXSMM_MALLOC_LOCK_PAGES)
      if (0 != (LIBXSMM_MALLOC_FLAG_PLOCK & local.flags)) { /* page-locked */
        LIBXSMM_ASSERT(0 != (LIBXSMM_MALLOC_FLAG_MMAP & local.flags));
        LIBXSMM_ATOMIC_SUB_FETCH(&internal_malloc_plocked, size_alloc, LIBXSMM_ATOMIC_RELAXED);
      }
# endif
#endif
      if (0 == (LIBXSMM_MALLOC_FLAG_PRIVATE & local.flags)) { /* public */
        if (0 != (LIBXSMM_MALLOC_FLAG_SCRATCH & local.flags)) { /* scratch */
          const size_t current = (size_t)LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_LOAD, LIBXSMM_BITS)(
            &internal_malloc_public_cur, LIBXSMM_ATOMIC_RELAXED);
          LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, LIBXSMM_BITS)(&internal_malloc_public_cur,
            size_alloc <= current ? (current - size_alloc) : 0, LIBXSMM_ATOMIC_RELAXED);
        }
        else { /* local */
          const size_t current = (size_t)LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_LOAD, LIBXSMM_BITS)(
            &internal_malloc_local_cur, LIBXSMM_ATOMIC_RELAXED);
          LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, LIBXSMM_BITS)(&internal_malloc_local_cur,
            size_alloc <= current ? (current - size_alloc) : 0, LIBXSMM_ATOMIC_RELAXED);
        }
      }
      else { /* private */
        const size_t current = (size_t)LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_LOAD, LIBXSMM_BITS)(
          &internal_malloc_private_cur, LIBXSMM_ATOMIC_RELAXED);
        LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, LIBXSMM_BITS)(&internal_malloc_private_cur,
          size_alloc <= current ? (current - size_alloc) : 0, LIBXSMM_ATOMIC_RELAXED);
      }
    }
  }
#if !defined(LIBXSMM_BUILD)
  else if ((LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity)
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM WARNING: attempt to release memory from non-matching implementation!\n");
  }
#endif
  return result;
}


LIBXSMM_API_INTERN void libxsmm_malloc_init(void)
{
#if (0 != LIBXSMM_SYNC) && defined(LIBXSMM_MALLOC_SCRATCH_JOIN)
  const char *const env = getenv("LIBXSMM_MALLOC_JOIN");
  if (NULL != env && '\0' != *env) internal_malloc_join = atoi(env);
#endif
#if defined(LIBXSMM_MALLOC_HOOK_DYNAMIC)
# if defined(LIBXSMM_MALLOC_HOOK_QKMALLOC)
  void* handle_qkmalloc = NULL;
  dlerror(); /* clear an eventual error status */
  handle_qkmalloc = dlopen("libqkmalloc.so", RTLD_LAZY);
  if (NULL != handle_qkmalloc) {
    libxsmm_malloc_fn.memalign.ptr = internal_memalign_malloc;
    libxsmm_malloc_fn.malloc.dlsym = dlsym(handle_qkmalloc, "malloc");
    if (NULL == dlerror() && NULL != libxsmm_malloc_fn.malloc.dlsym) {
#   if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
      libxsmm_malloc_fn.calloc.dlsym = dlsym(handle_qkmalloc, "calloc");
      if (NULL == dlerror() && NULL != libxsmm_malloc_fn.calloc.dlsym)
#   endif
      {
#   if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
        libxsmm_malloc_fn.realloc.dlsym = dlsym(handle_qkmalloc, "realloc");
        if (NULL == dlerror() && NULL != libxsmm_malloc_fn.realloc.dlsym)
#   endif
        {
          libxsmm_malloc_fn.free.dlsym = dlsym(handle_qkmalloc, "free");
        }
      }
    }
    dlclose(handle_qkmalloc);
  }
  if (NULL == libxsmm_malloc_fn.free.ptr)
# elif defined(LIBXSMM_MALLOC_HOOK_KMP)
  dlerror(); /* clear an eventual error status */
  libxsmm_malloc_fn.alignmem.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "kmp_aligned_malloc");
  if (NULL == dlerror() && NULL != libxsmm_malloc_fn.alignmem.dlsym) {
    libxsmm_malloc_fn.memalign.ptr = internal_memalign_twiddle;
    libxsmm_malloc_fn.malloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "kmp_malloc");
    if (NULL == dlerror() && NULL != libxsmm_malloc_fn.malloc.dlsym) {
# if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
      libxsmm_malloc_fn.calloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "kmp_calloc");
      if (NULL == dlerror() && NULL != libxsmm_malloc_fn.calloc.dlsym)
# endif
      {
# if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
        libxsmm_malloc_fn.realloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "kmp_realloc");
        if (NULL == dlerror() && NULL != libxsmm_malloc_fn.realloc.dlsym)
# endif
        {
          libxsmm_malloc_fn.free.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "kmp_free");
        }
      }
    }
  }
  if (NULL == libxsmm_malloc_fn.free.ptr)
# endif /*defined(LIBXSMM_MALLOC_HOOK_QKMALLOC)*/
  {
    dlerror(); /* clear an eventual error status */
# if (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD)))
    libxsmm_malloc_fn.memalign.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "__libc_memalign");
    if (NULL == dlerror() && NULL != libxsmm_malloc_fn.memalign.dlsym) {
      libxsmm_malloc_fn.malloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "__libc_malloc");
      if (NULL == dlerror() && NULL != libxsmm_malloc_fn.malloc.dlsym) {
#   if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
        libxsmm_malloc_fn.calloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "__libc_calloc");
        if (NULL == dlerror() && NULL != libxsmm_malloc_fn.calloc.dlsym)
#   endif
        {
#   if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
          libxsmm_malloc_fn.realloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "__libc_realloc");
          if (NULL == dlerror() && NULL != libxsmm_malloc_fn.realloc.dlsym)
#   endif
          {
            libxsmm_malloc_fn.free.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "__libc_free");
          }
        }
      }
    }
    if (NULL == libxsmm_malloc_fn.free.ptr) {
      void* handle_libc = NULL;
      dlerror(); /* clear an eventual error status */
      handle_libc = dlopen("libc.so." LIBXSMM_STRINGIFY(LIBXSMM_MALLOC_GLIBC), RTLD_LAZY);
      if (NULL != handle_libc) {
        libxsmm_malloc_fn.memalign.dlsym = dlsym(handle_libc, "__libc_memalign");
        if (NULL == dlerror() && NULL != libxsmm_malloc_fn.memalign.dlsym) {
          libxsmm_malloc_fn.malloc.dlsym = dlsym(handle_libc, "__libc_malloc");
          if (NULL == dlerror() && NULL != libxsmm_malloc_fn.malloc.dlsym) {
#   if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
            libxsmm_malloc_fn.calloc.dlsym = dlsym(handle_libc, "__libc_calloc");
            if (NULL == dlerror() && NULL != libxsmm_malloc_fn.calloc.dlsym)
#   endif
            {
#   if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
              libxsmm_malloc_fn.realloc.dlsym = dlsym(handle_libc, "__libc_realloc");
              if (NULL == dlerror() && NULL != libxsmm_malloc_fn.realloc.dlsym)
#   endif
              {
                libxsmm_malloc_fn.free.dlsym = dlsym(handle_libc, "__libc_free");
              }
            }
          }
        }
        dlclose(handle_libc);
      }
    }
#   if 0
    { /* attempt to setup deprecated GLIBC hooks */
      union { const void* dlsym; void* (**ptr)(size_t, size_t, const void*); } hook_memalign;
      dlerror(); /* clear an eventual error status */
      hook_memalign.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "__memalign_hook");
      if (NULL == dlerror() && NULL != hook_memalign.dlsym) {
        union { const void* dlsym; void* (**ptr)(size_t, const void*); } hook_malloc;
        hook_malloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "__malloc_hook");
        if (NULL == dlerror() && NULL != hook_malloc.dlsym) {
#   if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
          union { const void* dlsym; void* (**ptr)(void*, size_t, const void*); } hook_realloc;
          hook_realloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "__realloc_hook");
          if (NULL == dlerror() && NULL != hook_realloc.dlsym)
#   endif
          {
            union { const void* dlsym; void (**ptr)(void*, const void*); } hook_free;
            hook_free.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "__free_hook");
            if (NULL == dlerror() && NULL != hook_free.dlsym) {
              *hook_memalign.ptr = internal_memalign_hook;
              *hook_malloc.ptr = internal_malloc_hook;
#   if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
              *hook_realloc.ptr = internal_realloc_hook;
#   endif
              *hook_free.ptr = internal_free_hook;
            }
          }
        }
      }
    }
#   endif
# else /* TODO */
# endif /*(defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD)))*/
  }
  if (NULL != libxsmm_malloc_fn.free.ptr) {
# if defined(LIBXSMM_MALLOC_HOOK_IMALLOC)
    union { const void* dlsym; libxsmm_malloc_fun* ptr; } i_malloc;
    i_malloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "i_malloc");
    if (NULL == dlerror() && NULL != i_malloc.dlsym) {
#   if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
      union { const void* dlsym; void* (**ptr)(size_t, size_t); } i_calloc;
      i_calloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "i_calloc");
      if (NULL == dlerror() && NULL != i_calloc.dlsym)
#   endif
      {
#   if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
        union { const void* dlsym; libxsmm_realloc_fun* ptr; } i_realloc;
        i_realloc.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "i_realloc");
        if (NULL == dlerror() && NULL != i_realloc.dlsym)
#   endif
        {
          union { const void* dlsym; libxsmm_free_fun* ptr; } i_free;
          i_free.dlsym = dlsym(LIBXSMM_RTLD_NEXT, "i_free");
          if (NULL == dlerror() && NULL != i_free.dlsym) {
            *i_malloc.ptr = libxsmm_malloc_fn.malloc.ptr;
#   if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
            *i_calloc.ptr = libxsmm_malloc_fn.calloc.ptr;
#   endif
#   if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
            *i_realloc.ptr = libxsmm_malloc_fn.realloc.ptr;
#   endif
            *i_free.ptr = libxsmm_malloc_fn.free.ptr;
          }
        }
      }
    }
# endif /*defined(LIBXSMM_MALLOC_HOOK_IMALLOC)*/
  }
  else { /* fallback: potentially recursive */
# if (defined(LIBXSMM_BUILD) && (1 < (LIBXSMM_BUILD)))
    libxsmm_malloc_fn.memalign.ptr = __libc_memalign;
    libxsmm_malloc_fn.malloc.ptr = __libc_malloc;
#   if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
    libxsmm_malloc_fn.calloc.ptr = __libc_calloc;
#   endif
#   if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
    libxsmm_malloc_fn.realloc.ptr = __libc_realloc;
#   endif
    libxsmm_malloc_fn.free.ptr = __libc_free;
# else
    libxsmm_malloc_fn.memalign.ptr = libxsmm_memalign_internal;
    libxsmm_malloc_fn.malloc.ptr = malloc;
#   if defined(LIBXSMM_MALLOC_HOOK_CALLOC)
    libxsmm_malloc_fn.calloc.ptr = calloc;
#   endif
#   if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
    libxsmm_malloc_fn.realloc.ptr = realloc;
#   endif
    libxsmm_malloc_fn.free.ptr = free;
# endif
  }
#endif
}


LIBXSMM_API_INTERN void libxsmm_malloc_finalize(void)
{
}


LIBXSMM_API_INTERN int libxsmm_xset_default_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  const void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn)
{
  int result = EXIT_SUCCESS;
  if (NULL != lock) {
    LIBXSMM_INIT
    LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK, lock);
  }
  if (NULL != malloc_fn.function && NULL != free_fn.function) {
    libxsmm_default_allocator_context = context;
    libxsmm_default_malloc_fn = malloc_fn;
    libxsmm_default_free_fn = free_fn;
  }
  else {
    libxsmm_malloc_function internal_malloc_fn = { NULL };
    libxsmm_free_function internal_free_fn = { NULL };
    const void* internal_allocator = NULL;
#if defined(LIBXSMM_MALLOC_HOOK)
    internal_malloc_fn.function = __real_malloc;
    internal_free_fn.function = __real_free;
#else
    internal_malloc_fn.function = malloc;
    internal_free_fn.function = free;
#endif
    /*internal_allocator = NULL;*/
    if (NULL == malloc_fn.function && NULL == free_fn.function) {
      libxsmm_default_allocator_context = internal_allocator;
      libxsmm_default_malloc_fn = internal_malloc_fn;
      libxsmm_default_free_fn = internal_free_fn;
    }
    else { /* invalid allocator */
      static int error_once = 0;
      if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: allocator setup without malloc or free function!\n");
      }
      /* keep any valid (previously instantiated) default allocator */
      if (NULL == libxsmm_default_malloc_fn.function || NULL == libxsmm_default_free_fn.function) {
        libxsmm_default_allocator_context = internal_allocator;
        libxsmm_default_malloc_fn = internal_malloc_fn;
        libxsmm_default_free_fn = internal_free_fn;
      }
      result = EXIT_FAILURE;
    }
  }
  if (NULL != lock) {
    LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK, lock);
  }
  LIBXSMM_ASSERT(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_INTERN int libxsmm_xget_default_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  const void** context, libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn)
{
  int result = EXIT_SUCCESS;
  if (NULL != context || NULL != malloc_fn || NULL != free_fn) {
    if (NULL != lock) {
      LIBXSMM_INIT
      LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK, lock);
    }
    if (context) *context = libxsmm_default_allocator_context;
    if (NULL != malloc_fn) *malloc_fn = libxsmm_default_malloc_fn;
    if (NULL != free_fn) *free_fn = libxsmm_default_free_fn;
    if (NULL != lock) {
      LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK, lock);
    }
  }
  else if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM ERROR: invalid signature used to get the default memory allocator!\n");
    }
    result = EXIT_FAILURE;
  }
  LIBXSMM_ASSERT(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_INTERN int libxsmm_xset_scratch_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  const void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (NULL != lock) {
    LIBXSMM_INIT
    LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK, lock);
  }
  /* make sure the default allocator is setup before adopting it eventually */
  if (NULL == libxsmm_default_malloc_fn.function || NULL == libxsmm_default_free_fn.function) {
    const libxsmm_malloc_function null_malloc_fn = { NULL };
    const libxsmm_free_function null_free_fn = { NULL };
    libxsmm_xset_default_allocator(NULL/*already locked*/, NULL/*context*/, null_malloc_fn, null_free_fn);
  }
  if (NULL == malloc_fn.function && NULL == free_fn.function) { /* adopt default allocator */
    libxsmm_scratch_allocator_context = libxsmm_default_allocator_context;
    libxsmm_scratch_malloc_fn = libxsmm_default_malloc_fn;
    libxsmm_scratch_free_fn = libxsmm_default_free_fn;
  }
  else if (NULL != malloc_fn.function) {
    if (NULL == free_fn.function
      && /*warning*/(LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity)
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM WARNING: scratch allocator setup without free function!\n");
    }
    libxsmm_scratch_allocator_context = context;
    libxsmm_scratch_malloc_fn = malloc_fn;
    libxsmm_scratch_free_fn = free_fn; /* NULL allowed */
  }
  else { /* invalid scratch allocator */
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: invalid scratch allocator (default used)!\n");
    }
    /* keep any valid (previously instantiated) scratch allocator */
    if (NULL == libxsmm_scratch_malloc_fn.function) {
      libxsmm_scratch_allocator_context = libxsmm_default_allocator_context;
      libxsmm_scratch_malloc_fn = libxsmm_default_malloc_fn;
      libxsmm_scratch_free_fn = libxsmm_default_free_fn;
    }
    result = EXIT_FAILURE;
  }
  if (NULL != lock) {
    LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK, lock);
  }
  LIBXSMM_ASSERT(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_INTERN int libxsmm_xget_scratch_allocator(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock,
  const void** context, libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn)
{
  int result = EXIT_SUCCESS;
  if (NULL != context || NULL != malloc_fn || NULL != free_fn) {
    if (NULL != lock) {
      LIBXSMM_INIT
      LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK, lock);
    }
    if (context) *context = libxsmm_scratch_allocator_context;
    if (NULL != malloc_fn) *malloc_fn = libxsmm_scratch_malloc_fn;
    if (NULL != free_fn) *free_fn = libxsmm_scratch_free_fn;
    if (NULL != lock) {
      LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK, lock);
    }
  }
  else if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM ERROR: invalid signature used to get the scratch memory allocator!\n");
    }
    result = EXIT_FAILURE;
  }
  LIBXSMM_ASSERT(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API int libxsmm_set_default_allocator(const void* context,
  libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn)
{
  return libxsmm_xset_default_allocator(&libxsmm_lock_global, context, malloc_fn, free_fn);
}


LIBXSMM_API int libxsmm_get_default_allocator(const void** context,
  libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn)
{
  return libxsmm_xget_default_allocator(&libxsmm_lock_global, context, malloc_fn, free_fn);
}


LIBXSMM_API int libxsmm_set_scratch_allocator(const void* context,
  libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn)
{
  return libxsmm_xset_scratch_allocator(&libxsmm_lock_global, context, malloc_fn, free_fn);
}


LIBXSMM_API int libxsmm_get_scratch_allocator(const void** context,
  libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn)
{
  return libxsmm_xget_scratch_allocator(&libxsmm_lock_global, context, malloc_fn, free_fn);
}


LIBXSMM_API int libxsmm_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra)
{
  int result;
#if !defined(NDEBUG)
  if (NULL != size || NULL != extra)
#endif
  {
    const int check = ((NULL == flags || 0 == (LIBXSMM_MALLOC_FLAG_X & *flags)) ? 2 : 1);
    const internal_malloc_info_type *const info = internal_malloc_info(memory, check);
    if (NULL != info) {
      if (NULL != size) *size = info->size;
      if (NULL != flags) *flags = info->flags;
      if (NULL != extra) *extra = info->pointer;
      result = EXIT_SUCCESS;
    }
    else { /* potentially foreign buffer */
      result = (NULL != memory ? EXIT_FAILURE : EXIT_SUCCESS);
      if (NULL != size) *size = 0;
      if (NULL != flags) *flags = 0;
      if (NULL != extra) *extra = NULL;
    }
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: attachment error for memory buffer %p!\n", memory);
    }
    LIBXSMM_ASSERT_MSG(0/*false*/, "LIBXSMM ERROR: attachment error");
    result = EXIT_FAILURE;
  }
#endif
  return result;
}


#if !defined(_WIN32)

LIBXSMM_API_INLINE void internal_xmalloc_mhint(void* buffer, size_t size)
{
  LIBXSMM_ASSERT((MAP_FAILED != buffer && NULL != buffer) || 0 == size);
#if (defined(_DEFAULT_SOURCE) || defined(_BSD_SOURCE))
  /* proceed after failed madvise (even in case of an error; take what we got) */
  /* issue no warning as a failure seems to be related to the kernel version */
  madvise(buffer, size, MADV_NORMAL/*MADV_RANDOM*/
# if defined(MADV_NOHUGEPAGE) /* if not available, we then take what we got (THP) */
    | ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) > size ? MADV_NOHUGEPAGE : 0)
# endif
# if defined(MADV_DONTDUMP)
    | ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) > size ? 0 : MADV_DONTDUMP)
# endif
  );
#else
  LIBXSMM_UNUSED(buffer); LIBXSMM_UNUSED(size);
#endif
}


LIBXSMM_API_INLINE void* internal_xmalloc_xmap(const char* dir, size_t size, int flags, void** rx)
{
  void* result = MAP_FAILED;
  char filename[4096] = LIBXSMM_MALLOC_XMAP_TEMPLATE;
  int i = 0;
  LIBXSMM_ASSERT(NULL != rx && MAP_FAILED != *rx);
  if (NULL != dir && '\0' != *dir) {
    i = LIBXSMM_SNPRINTF(filename, sizeof(filename), "%s/" LIBXSMM_MALLOC_XMAP_TEMPLATE, dir);
  }
  if (0 <= i && i < (int)sizeof(filename)) {
    /* coverity[secure_temp] */
    i = LIBXSMM_MKTEMP(filename);
    if (0 <= i) {
      if (0 == unlink(filename) && 0 == ftruncate(i, size) /*&& 0 == chmod(filename, S_IRWXU)*/) {
        const int mflags = (flags | LIBXSMM_MAP_SHARED | LIBXSMM_MAP_JIT);
        void *const xmap = mmap(*rx, size, PROT_READ | PROT_EXEC, mflags, i, 0/*offset*/);
        if (MAP_FAILED != xmap) {
          LIBXSMM_ASSERT(NULL != xmap);
#if defined(MAP_32BIT)
          result = mmap(NULL, size, PROT_READ | PROT_WRITE, mflags & ~MAP_32BIT, i, 0/*offset*/);
#else
          result = mmap(NULL, size, PROT_READ | PROT_WRITE, mflags, i, 0/*offset*/);
#endif
          if (MAP_FAILED != result) {
            LIBXSMM_ASSERT(NULL != result);
            internal_xmalloc_mhint(xmap, size);
            *rx = xmap;
          }
          else {
            munmap(xmap, size);
            *rx = NULL;
          }
        }
      }
      close(i);
    }
  }
  return result;
}

#endif /*!defined(_WIN32)*/


LIBXSMM_API_INLINE void* internal_xrealloc(void** ptr, internal_malloc_info_type** info, size_t size,
  libxsmm_realloc_fun realloc_fn, libxsmm_free_fun free_fn)
{
  char *const base = (char*)(NULL != *info ? (*info)->pointer : *ptr), *result;
  LIBXSMM_ASSERT(NULL != *ptr && NULL != free_fn);
  /* reallocation may implicitly invalidate info */
  result = (char*)(NULL != realloc_fn ? realloc_fn(base, size) : __real_malloc(size));
  if (result == base) { /* signal no-copy */
    LIBXSMM_ASSERT(NULL != result);
    *info = NULL; /* no delete */
    *ptr = NULL; /* no copy */
  }
  else if (NULL != result) { /* copy */
    if (NULL != realloc_fn) {
      const size_t offset_src = (const char*)*ptr - base;
      *ptr = result + offset_src; /* copy */
      *info = NULL; /* no delete */
    }
  }
#if !defined(NDEBUG) && 0
  else { /* failed */
    if (NULL != *info) {
      internal_xfree(*ptr, *info); /* invalidates info */
    }
    else { /* foreign pointer */
      free_fn(*ptr);
    }
    *info = NULL; /* no delete */
    *ptr = NULL; /* no copy */
  }
#else
  LIBXSMM_UNUSED(free_fn);
#endif
  return result;
}


LIBXSMM_API_INTERN void* internal_xmalloc(void** /*ptr*/, internal_malloc_info_type** /*info*/, size_t /*size*/,
  const void* /*context*/, libxsmm_malloc_function /*malloc_fn*/, libxsmm_free_function /*free_fn*/);
LIBXSMM_API_INTERN void* internal_xmalloc(void** ptr, internal_malloc_info_type** info, size_t size,
  const void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn)
{
  void* result;
  LIBXSMM_ASSERT(NULL != ptr && NULL != info && NULL != malloc_fn.function);
  if (NULL == *ptr) {
    result = (NULL == context
      ? malloc_fn.function(size)
      : malloc_fn.ctx_form(size, context));
  }
  else { /* reallocate */
    if (NULL != free_fn.function /* prefer free_fn since it is part of pointer-info */
      ? (__real_free == free_fn.function || free == free_fn.function)
      : (__real_malloc == malloc_fn.function || malloc == malloc_fn.function))
    {
#if defined(LIBXSMM_MALLOC_HOOK_REALLOC)
      result = internal_xrealloc(ptr, info, size, __real_realloc, __real_free);
#else
      result = internal_xrealloc(ptr, info, size, NULL, __real_free);
#endif
    }
    else { /* fallback with regular allocation */
      result = (NULL == context
        ? malloc_fn.function(size)
        : malloc_fn.ctx_form(size, context));
      if (NULL == result) { /* failed */
        if (NULL != *info) {
          internal_xfree(*ptr, *info); /* invalidates info */
        }
        else { /* foreign pointer */
          (NULL != free_fn.function ? free_fn.function : __real_free)(*ptr);
        }
        *ptr = NULL; /* safe delete */
      }
    }
  }
  return result;
}


LIBXSMM_API int libxsmm_xmalloc(void** memory, size_t size, size_t alignment,
  int flags, const void* extra, size_t extra_size)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG)
  if (NULL != memory)
#endif
  {
    static int error_once = 0;
    if (0 != size) {
      size_t alloc_alignment = 0, alloc_size = 0, max_preserve = 0;
      internal_malloc_info_type* info = NULL;
      void *buffer = NULL, *reloc = NULL;
      /* ATOMIC BEGIN: this region should be atomic/locked */
      const void* context = libxsmm_default_allocator_context;
      libxsmm_malloc_function malloc_fn = libxsmm_default_malloc_fn;
      libxsmm_free_function free_fn = libxsmm_default_free_fn;
      if (0 != (LIBXSMM_MALLOC_FLAG_SCRATCH & flags)) {
        context = libxsmm_scratch_allocator_context;
        malloc_fn = libxsmm_scratch_malloc_fn;
        free_fn = libxsmm_scratch_free_fn;
#if defined(LIBXSMM_MALLOC_MMAP_SCRATCH)
        flags |= LIBXSMM_MALLOC_FLAG_MMAP;
#endif
      }
      if ((0 != (internal_malloc_kind & 1) && 0 < internal_malloc_kind)
        || NULL == malloc_fn.function || NULL == free_fn.function)
      {
        malloc_fn.function = __real_malloc;
        free_fn.function = __real_free;
        context = NULL;
      }
      /* ATOMIC END: this region should be atomic */
      flags |= LIBXSMM_MALLOC_FLAG_RW; /* normalize given flags since flags=0 is accepted as well */
      if (0 != (LIBXSMM_MALLOC_FLAG_REALLOC & flags) && NULL != *memory) {
        info = internal_malloc_info(*memory, 2/*check*/);
        if (NULL != info) {
          max_preserve = info->size;
        }
        else { /* reallocation of unknown allocation */
          flags &= ~LIBXSMM_MALLOC_FLAG_MMAP;
        }
      }
      else *memory = NULL;
#if !defined(LIBXSMM_MALLOC_MMAP)
      if (0 == (LIBXSMM_MALLOC_FLAG_X & flags) && 0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) {
        alloc_alignment = (0 == (LIBXSMM_MALLOC_FLAG_REALLOC & flags) ? libxsmm_alignment(size, alignment) : alignment);
        alloc_size = size + extra_size + sizeof(internal_malloc_info_type) + alloc_alignment - 1;
        buffer = internal_xmalloc(memory, &info, alloc_size, context, malloc_fn, free_fn);
      }
      else
#endif
      if (NULL == info || size != info->size) {
#if defined(_WIN32) || defined(__CYGWIN__)
        const int mflags = (0 != (LIBXSMM_MALLOC_FLAG_X & flags) ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
        static SIZE_T alloc_alignmax = 0, alloc_pagesize = 0;
        if (0 == alloc_alignmax) { /* first/one time */
          SYSTEM_INFO system_info;
          GetSystemInfo(&system_info);
          alloc_pagesize = system_info.dwPageSize;
          alloc_alignmax = GetLargePageMinimum();
        }
        if ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) <= size) { /* attempt to use large pages */
          HANDLE process_token;
          alloc_alignment = (NULL == info
            ? (0 == alignment ? alloc_alignmax : libxsmm_lcm(alignment, alloc_alignmax))
            : libxsmm_lcm(alignment, alloc_alignmax));
          alloc_size = LIBXSMM_UP2(size + extra_size + sizeof(internal_malloc_info_type) + alloc_alignment - 1, alloc_alignmax);
          if (TRUE == OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &process_token)) {
            TOKEN_PRIVILEGES tp = { 0 };
            if (TRUE == LookupPrivilegeValue(NULL, TEXT("SeLockMemoryPrivilege"), &tp.Privileges[0].Luid)) {
              tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED; tp.PrivilegeCount = 1; /* enable privilege */
              if (TRUE == AdjustTokenPrivileges(process_token, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0)
                && ERROR_SUCCESS == GetLastError()/*may has failed (regardless of TRUE)*/)
              {
                /* VirtualAlloc cannot be used to reallocate memory */
                buffer = VirtualAlloc(NULL, alloc_size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, mflags);
              }
              tp.Privileges[0].Attributes = 0; /* disable privilege */
              AdjustTokenPrivileges(process_token, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
            }
            CloseHandle(process_token);
          }
        }
        else { /* small allocation using regular page-size */
          alloc_alignment = (NULL == info ? libxsmm_alignment(size, alignment) : alignment);
          alloc_size = LIBXSMM_UP2(size + extra_size + sizeof(internal_malloc_info_type) + alloc_alignment - 1, alloc_pagesize);
        }
        if (NULL == buffer) { /* small allocation or retry with regular page size */
          /* VirtualAlloc cannot be used to reallocate memory */
          buffer = VirtualAlloc(NULL, alloc_size, MEM_RESERVE | MEM_COMMIT, mflags);
        }
        if (NULL != buffer) {
          flags |= LIBXSMM_MALLOC_FLAG_MMAP; /* select the corresponding deallocation */
        }
        else if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) { /* fallback allocation */
          buffer = internal_xmalloc(memory, &info, alloc_size, context, malloc_fn, free_fn);
        }
#else /* !defined(_WIN32) */
# if defined(MAP_HUGETLB) && defined(LIBXSMM_MALLOC_HUGE_PAGES)
        static size_t limit_hugetlb = LIBXSMM_SCRATCH_UNLIMITED;
# endif
# if defined(MAP_LOCKED) && defined(LIBXSMM_MALLOC_LOCK_PAGES)
        static size_t limit_plocked = LIBXSMM_SCRATCH_UNLIMITED;
# endif
# if defined(MAP_32BIT)
        static int map32 = 1;
# endif
        int mflags = 0
# if defined(MAP_UNINITIALIZED) && 0/*fails with WSL*/
          | MAP_UNINITIALIZED /* unlikely available */
# endif
# if defined(MAP_NORESERVE)
          | (LIBXSMM_MALLOC_ALIGNMAX < size ? 0 : MAP_NORESERVE)
# endif
# if defined(MAP_32BIT)
          | ((0 != (LIBXSMM_MALLOC_FLAG_X & flags) && 0 != map32) ? MAP_32BIT : 0)
# endif
# if defined(MAP_HUGETLB) && defined(LIBXSMM_MALLOC_HUGE_PAGES)
          | ((0 == (LIBXSMM_MALLOC_FLAG_X & flags)
            && ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) <= size ||
              0 != (LIBXSMM_MALLOC_FLAG_PHUGE & flags))
            && (internal_malloc_hugetlb + size) < limit_hugetlb) ? MAP_HUGETLB : 0)
# endif
# if defined(MAP_LOCKED) && defined(LIBXSMM_MALLOC_LOCK_PAGES) && 0 == (LIBXSMM_MALLOC_LOCK_PAGES)
          | (((0 != (LIBXSMM_MALLOC_FLAG_PLOCK & flags) || 0 == (LIBXSMM_MALLOC_FLAG_X & flags))
            && (internal_malloc_plocked + size) < limit_plocked) ? MAP_LOCKED : 0)
# endif
          | (0 != (LIBXSMM_MALLOC_FLAG_X & flags) ? LIBXSMM_MAP_JIT : 0)
        ; /* mflags */
# if defined(MAP_POPULATE)
        { static int prefault = 0;
          if (0 == prefault) { /* prefault only on Linux 3.10.0-327 (and later) to avoid data race in page-fault handler */
            struct utsname osinfo; unsigned int version_major = 3, version_minor = 10, version_update = 0, version_patch = 327;
            if (0 <= uname(&osinfo) && 0 == strcmp("Linux", osinfo.sysname)
              && 4 == sscanf(osinfo.release, "%u.%u.%u-%u", &version_major, &version_minor, &version_update, &version_patch)
              && LIBXSMM_VERSION4(3, 10, 0, 327) > LIBXSMM_VERSION4(version_major, version_minor, version_update, version_patch))
            {
              mflags |= MAP_POPULATE; prefault = 1;
            }
            else prefault = -1;
          }
          else if (1 == prefault) mflags |= MAP_POPULATE;
        }
# endif
        /* make allocated size at least a multiple of the smallest page-size to avoid split-pages (unmap!) */
        alloc_alignment = libxsmm_lcm(0 == alignment ? libxsmm_alignment(size, alignment) : alignment, LIBXSMM_PAGE_MINSIZE);
        alloc_size = LIBXSMM_UP2(size + extra_size + sizeof(internal_malloc_info_type) + alloc_alignment - 1, alloc_alignment);
        if (0 == (LIBXSMM_MALLOC_FLAG_X & flags)) { /* anonymous and non-executable */
# if defined(MAP_32BIT)
          LIBXSMM_ASSERT(0 == (MAP_32BIT & mflags));
# endif
# if 0
          LIBXSMM_ASSERT(NULL != info || NULL == *memory); /* no memory mapping of foreign pointer */
# endif
          buffer = mmap(NULL == info ? NULL : info->pointer, alloc_size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | LIBXSMM_MAP_ANONYMOUS | mflags, -1, 0/*offset*/);
# if defined(MAP_HUGETLB) && defined(LIBXSMM_MALLOC_HUGE_PAGES)
          INTERNAL_XMALLOC_KIND(MAP_HUGETLB, "huge-page", LIBXSMM_MALLOC_FLAG_PHUGE, flags, mflags,
            internal_malloc_hugetlb, limit_hugetlb, info, alloc_size, buffer);
# endif
# if defined(MAP_LOCKED) && defined(LIBXSMM_MALLOC_LOCK_PAGES)
#   if 0 == (LIBXSMM_MALLOC_LOCK_PAGES)
          INTERNAL_XMALLOC_KIND(MAP_LOCKED, "locked-page", LIBXSMM_MALLOC_FLAG_PLOCK, flags, mflags,
            internal_malloc_plocked, limit_plocked, info, alloc_size, buffer);
#   else
          if (0 != (MAP_LOCKED & mflags) && MAP_FAILED != buffer) {
            LIBXSMM_ASSERT(NULL != buffer);
#     if 1 == (LIBXSMM_MALLOC_LOCK_PAGES) || !defined(MLOCK_ONFAULT) || !defined(SYS_mlock2)
            if (0 == mlock(buffer, alloc_size))
#     elif 0 /* mlock2 is potentially not exposed */
            if (0 == mlock2(buffer, alloc_size, MLOCK_ONFAULT))
#     else
            if (0 == syscall(SYS_mlock2, buffer, alloc_size, MLOCK_ONFAULT))
#     endif
            {
              LIBXSMM_ATOMIC_ADD_FETCH(&internal_malloc_plocked, alloc_size, LIBXSMM_ATOMIC_RELAXED);
              flags |= LIBXSMM_MALLOC_FLAG_PLOCK;
            }
            else { /* update watermark */
              INTERNAL_XMALLOC_WATERMARK("locked-page", internal_malloc_plocked, limit_plocked, alloc_size);
              flags &= ~LIBXSMM_MALLOC_FLAG_PLOCK;
            }
          }
#   endif
# endif
        }
        else { /* executable buffer requested */
          static /*LIBXSMM_TLS*/ int entrypoint = -1; /* fallback allocation method */
# if defined(MAP_HUGETLB) && defined(LIBXSMM_MALLOC_HUGE_PAGES)
          LIBXSMM_ASSERT(0 == (MAP_HUGETLB & mflags));
# endif
# if defined(MAP_LOCKED) && defined(LIBXSMM_MALLOC_LOCK_PAGES)
          LIBXSMM_ASSERT(0 == (MAP_LOCKED & mflags));
# endif
          if (0 > (int)LIBXSMM_ATOMIC_LOAD(&entrypoint, LIBXSMM_ATOMIC_RELAXED)) {
            const char *const env = getenv("LIBXSMM_SE");
            LIBXSMM_ATOMIC_STORE(&entrypoint, NULL == env
              /* libxsmm_se decides */
              ? (0 == libxsmm_se ? LIBXSMM_MALLOC_FINAL : LIBXSMM_MALLOC_FALLBACK)
              /* user's choice takes precedence */
              : ('0' != *env ? LIBXSMM_MALLOC_FALLBACK : LIBXSMM_MALLOC_FINAL),
              LIBXSMM_ATOMIC_SEQ_CST);
            LIBXSMM_ASSERT(0 <= entrypoint);
          }
          INTERNAL_XMALLOC(0, entrypoint, "JITDUMPDIR", "", map32, mflags, alloc_size, buffer, &reloc); /* 1st try */
          INTERNAL_XMALLOC(1, entrypoint, "TMPDIR", "/tmp", map32, mflags, alloc_size, buffer, &reloc); /* 2nd try */
          /* coverity[string_size] */
          INTERNAL_XMALLOC(2, entrypoint, "HOME", "", map32, mflags, alloc_size, buffer, &reloc); /* 3rd try */
          if (3 >= entrypoint && (MAP_FAILED == buffer || NULL == buffer)) { /* 4th try */
            buffer = mmap(reloc, alloc_size, PROT_READ | PROT_WRITE | PROT_EXEC,
# if defined(MAP_32BIT)
              MAP_PRIVATE | LIBXSMM_MAP_ANONYMOUS | (0 == map32 ? (mflags & ~MAP_32BIT) : mflags),
# else
              MAP_PRIVATE | LIBXSMM_MAP_ANONYMOUS | mflags,
# endif
              -1, 0/*offset*/);
            if (MAP_FAILED != buffer) entrypoint = 3;
# if defined(MAP_32BIT)
            else if (0 != (MAP_32BIT & mflags) && 0 != map32) {
              buffer = mmap(reloc, alloc_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | LIBXSMM_MAP_ANONYMOUS | (mflags & ~MAP_32BIT),
                - 1, 0/*offset*/);
              if (MAP_FAILED != buffer) {
                entrypoint = 3;
                map32 = 0;
              }
            }
# endif
          }
          /* upgrade to SE-mode and retry lower entry-points */
          if (MAP_FAILED == buffer && 0 == libxsmm_se) {
            libxsmm_se = 1; entrypoint = 0;
            INTERNAL_XMALLOC(0, entrypoint, "JITDUMPDIR", "", map32, mflags, alloc_size, buffer, &reloc); /* 1st try */
            INTERNAL_XMALLOC(1, entrypoint, "TMPDIR", "/tmp", map32, mflags, alloc_size, buffer, &reloc); /* 2nd try */
            INTERNAL_XMALLOC(2, entrypoint, "HOME", "", map32, mflags, alloc_size, buffer, &reloc); /* 3rd try */
          }
        }
        if (MAP_FAILED != buffer && NULL != buffer) {
          flags |= LIBXSMM_MALLOC_FLAG_MMAP; /* select deallocation */
        }
        else { /* allocation failed */
          if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) { /* ultimate fallback */
            buffer = (NULL != malloc_fn.function
              ? (NULL == context ? malloc_fn.function(alloc_size) : malloc_fn.ctx_form(alloc_size, context))
              : (NULL));
          }
          reloc = NULL;
        }
        if (MAP_FAILED != buffer && NULL != buffer) {
          internal_xmalloc_mhint(buffer, alloc_size);
# if defined(__APPLE__) && defined(__arm64__)
          if (0 != (LIBXSMM_MALLOC_FLAG_W & flags)) {
            pthread_jit_write_protect_np(0/*false*/);
          }
# endif
        }
#endif /* !defined(_WIN32) */
      }
      else { /* reallocation of the same pointer and size */
        alloc_size = size + extra_size + sizeof(internal_malloc_info_type) + alignment - 1;
        if (NULL != info) {
          buffer = info->pointer;
          flags |= info->flags;
        }
        else {
          flags |= LIBXSMM_MALLOC_FLAG_MMAP;
          buffer = *memory;
        }
        alloc_alignment = alignment;
        *memory = NULL; /* signal no-copy */
      }
      if (
#if !defined(_WIN32) && !defined(__clang_analyzer__)
        MAP_FAILED != buffer &&
#endif
        NULL != buffer)
      {
        char *const cbuffer = (char*)buffer, *const aligned = LIBXSMM_ALIGN(
          cbuffer + extra_size + sizeof(internal_malloc_info_type), alloc_alignment);
        internal_malloc_info_type *const buffer_info = (internal_malloc_info_type*)(
          aligned - sizeof(internal_malloc_info_type));
        LIBXSMM_ASSERT((aligned + size) <= (cbuffer + alloc_size));
        LIBXSMM_ASSERT(0 < alloc_alignment);
        /* former content must be preserved prior to setup of buffer_info */
        if (NULL != *memory) { /* preserve/copy previous content */
#if 0
          LIBXSMM_ASSERT(0 != (LIBXSMM_MALLOC_FLAG_REALLOC & flags));
#endif
          /* content behind foreign pointers is not explicitly preserved; buffers may overlap */
          memmove(aligned, *memory, LIBXSMM_MIN(max_preserve, size));
          if (NULL != info /* known allocation (non-foreign pointer) */
            && EXIT_SUCCESS != internal_xfree(*memory, info) /* !libxsmm_free, invalidates info */
            && 0 != libxsmm_verbosity /* library code is expected to be mute */
            && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
          { /* display some extra context of the failure (reallocation) */
            fprintf(stderr, "LIBXSMM ERROR: memory reallocation failed to release memory!\n");
          }
        }
        if (NULL != extra || 0 == extra_size) {
          const char *const src = (const char*)extra;
          int i; for (i = 0; i < (int)extra_size; ++i) cbuffer[i] = src[i];
        }
        else if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: incorrect extraneous data specification!\n");
          /* no EXIT_FAILURE because valid buffer is returned */
        }
        if (0 == (LIBXSMM_MALLOC_FLAG_X & flags)) { /* update statistics */
          if (0 == (LIBXSMM_MALLOC_FLAG_PRIVATE & flags)) { /* public */
            if (0 != (LIBXSMM_MALLOC_FLAG_SCRATCH & flags)) { /* scratch */
              const size_t watermark = LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_ADD_FETCH, LIBXSMM_BITS)(
                &internal_malloc_public_cur, alloc_size, LIBXSMM_ATOMIC_RELAXED);
              if (internal_malloc_public_max < watermark) internal_malloc_public_max = watermark; /* accept data-race */
            }
            else { /* local */
              const size_t watermark = LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_ADD_FETCH, LIBXSMM_BITS)(
                &internal_malloc_local_cur, alloc_size, LIBXSMM_ATOMIC_RELAXED);
              if (internal_malloc_local_max < watermark) internal_malloc_local_max = watermark; /* accept data-race */
            }
          }
          else if (0 != (LIBXSMM_MALLOC_FLAG_SCRATCH & flags)) { /* private scratch */
            const size_t watermark = LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_ADD_FETCH, LIBXSMM_BITS)(
              &internal_malloc_private_cur, alloc_size, LIBXSMM_ATOMIC_RELAXED);
            if (internal_malloc_private_max < watermark) internal_malloc_private_max = watermark; /* accept data-race */
          }
        }
        /* keep allocation function on record */
        if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) {
          buffer_info->context = context;
          buffer_info->free = free_fn;
        }
        else {
          buffer_info->free.function = NULL;
          buffer_info->context = NULL;
        }
#if defined(LIBXSMM_MALLOC_INFO_ALLOCSIZE)
        buffer_info->size_alloc = alloc_size;
#endif
        buffer_info->size = size;
        buffer_info->pointer = buffer;
        buffer_info->reloc = reloc;
        buffer_info->flags = flags;
#if defined(LIBXSMM_VTUNE)
        buffer_info->code_id = 0;
#endif /* info must be initialized to calculate correct checksum */
#if !defined(LIBXSMM_MALLOC_CRC_OFF)
# if defined(LIBXSMM_MALLOC_CRC_LIGHT)
        buffer_info->hash = LIBXSMM_CRCPTR(LIBXSMM_MALLOC_SEED, buffer_info);
# else
        buffer_info->hash = libxsmm_crc32(LIBXSMM_MALLOC_SEED, buffer_info,
          (unsigned int)(((char*)&buffer_info->hash) - ((char*)buffer_info)));
# endif
#endif  /* finally commit/return allocated buffer */
        *memory = aligned;
      }
      else {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
         && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          char alloc_size_buffer[32];
          libxsmm_format_value(alloc_size_buffer, sizeof(alloc_size_buffer), alloc_size, "KM", "B", 10);
          fprintf(stderr, "LIBXSMM ERROR: failed to allocate %s with flag=%i!\n", alloc_size_buffer, flags);
        }
        result = EXIT_FAILURE;
        *memory = NULL;
      }
    }
    else {
      if ((LIBXSMM_VERBOSITY_HIGH <= libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM WARNING: zero-sized memory allocation detected!\n");
      }
      *memory = NULL; /* no EXIT_FAILURE */
    }
  }
#if !defined(NDEBUG)
  else if (0 != size) {
    result = EXIT_FAILURE;
  }
#endif
  return result;
}


LIBXSMM_API void libxsmm_xfree(const void* memory, int check)
{
#if (!defined(LIBXSMM_MALLOC_HOOK) || defined(_DEBUG))
  static int error_once = 0;
#endif
  /*const*/ internal_malloc_info_type *const info = internal_malloc_info(memory, check);
  if (NULL != info) { /* !libxsmm_free */
#if (!defined(LIBXSMM_MALLOC_HOOK) || defined(_DEBUG))
    if (EXIT_SUCCESS != internal_xfree(memory, info)) { /* invalidates info */
      if ( 0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: memory deallocation failed!\n");
      }
    }
#else
    internal_xfree(memory, info); /* invalidates info */
#endif
  }
  else if (NULL != memory) {
#if 1
    union { const void* const_ptr; void* ptr; } cast = { 0 };
    cast.const_ptr = memory; /* C-cast still warns */
    __real_free(cast.ptr);
#endif
#if (!defined(LIBXSMM_MALLOC_HOOK) || defined(_DEBUG))
    if ( 0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: deallocation does not match allocation!\n");
    }
#endif
  }
}


#if defined(LIBXSMM_VTUNE)
LIBXSMM_API_INLINE void internal_get_vtune_jitdesc(const void* code,
  unsigned int code_id, size_t code_size, const char* code_name,
  LIBXSMM_VTUNE_JIT_DESC_TYPE* desc)
{
  LIBXSMM_ASSERT(NULL != code && 0 != code_id && 0 != code_size && NULL != desc);
  desc->method_id = code_id;
  /* incorrect constness (method_name) */
  desc->method_name = (char*)code_name;
  /* incorrect constness (method_load_address) */
  desc->method_load_address = (void*)code;
  desc->method_size = code_size;
  desc->line_number_size = 0;
  desc->line_number_table = NULL;
  desc->class_file_name = NULL;
  desc->source_file_name = NULL;
# if (2 <= LIBXSMM_VTUNE_JITVERSION)
  desc->module_name = "libxsmm.jit";
# endif
}
#endif


LIBXSMM_API_INTERN int libxsmm_malloc_xattrib(void* buffer, int flags, size_t size)
{
  int result = EXIT_SUCCESS;
#if defined(_WIN32)
  LIBXSMM_ASSERT(NULL != buffer || 0 == size);
#else
  LIBXSMM_ASSERT((NULL != buffer && MAP_FAILED != buffer) || 0 == size);
#endif
  /* quietly keep the read permission, but eventually revoke write permissions */
  if (0 == (LIBXSMM_MALLOC_FLAG_W & flags) || 0 != (LIBXSMM_MALLOC_FLAG_X & flags)) {
    if (0 == (LIBXSMM_MALLOC_FLAG_X & flags)) { /* data-buffer; non-executable */
#if defined(_WIN32)
      /* TODO: implement memory protection under Microsoft Windows */
#else
      const int result_mprotect = mprotect(buffer, size, PROT_READ);
# if defined(__APPLE__) && defined(__arm64__)
      if (EXIT_SUCCESS != result_mprotect) {
        static int error_once = 0;
        if ((LIBXSMM_VERBOSITY_HIGH <= libxsmm_verbosity || 0 > libxsmm_verbosity)
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM WARNING: failed to mark buffer as read-only!\n");
        }
      }
# else
      result = result_mprotect;
# endif
#endif
    }
    else { /* executable buffer requested */
#if defined(_WIN32)
      /* TODO: implement memory protection under Microsoft Windows */
#else /* treat memory protection errors as soft error; ignore return value */
# if defined(__APPLE__) && defined(__arm64__)
      if (0 == (LIBXSMM_MALLOC_FLAG_W & flags)) {
        pthread_jit_write_protect_np(1/*true*/);
      }
# else
      const int result_mprotect = mprotect(buffer, size, PROT_READ | PROT_EXEC);
      if (EXIT_SUCCESS != result_mprotect) {
        static int error_once = 0;
        if (0 != libxsmm_se) { /* hard-error in case of SELinux */
          if (0 != libxsmm_verbosity /* library code is expected to be mute */
            && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXSMM ERROR: failed to allocate an executable buffer!\n");
          }
          result = result_mprotect;
        }
        else if ((LIBXSMM_VERBOSITY_HIGH <= libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM WARNING: read-only request for JIT-buffer failed!\n");
        }
      }
# endif
#endif
    }
  }
  return result;
}


LIBXSMM_API_INTERN int libxsmm_malloc_attrib(void** memory, int flags, const char* name, const size_t* data_size)
{
  internal_malloc_info_type *const info = (NULL != memory ? internal_malloc_info(*memory, 0/*no check*/) : NULL);
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (NULL != info) {
    const size_t info_size = info->size, apply_size = (NULL == data_size
      ? info_size : (info_size - LIBXSMM_MIN(*data_size, info_size)));
    void *const buffer = info->pointer;
#if defined(_WIN32)
    LIBXSMM_ASSERT(NULL != buffer || 0 == apply_size);
#else
    LIBXSMM_ASSERT((NULL != buffer && MAP_FAILED != buffer) || 0 == apply_size);
#endif
    flags |= (info->flags & ~LIBXSMM_MALLOC_FLAG_RWX); /* merge with current flags */
    /* quietly keep the read permission, but eventually revoke write permissions */
    if (0 == (LIBXSMM_MALLOC_FLAG_W & flags) || 0 != (LIBXSMM_MALLOC_FLAG_X & flags)) {
      const size_t alignment = (size_t)(((const char*)(*memory)) - ((const char*)buffer));
      const size_t alloc_size = apply_size + alignment;
      if (0 == (LIBXSMM_MALLOC_FLAG_X & flags)) { /* data-buffer; non-executable */
        result = libxsmm_malloc_xattrib(buffer, flags, alloc_size);
      }
      else { /* executable buffer requested */
        void *const code_ptr = (NULL != info->reloc ? ((void*)(((char*)info->reloc) + alignment)) : *memory);
        LIBXSMM_ASSERT(0 != (LIBXSMM_MALLOC_FLAG_X & flags));
        if (NULL != name && '\0' != *name) { /* profiler support requested */
          if (0 > libxsmm_verbosity) { /* avoid dump if just the profiler is enabled */
            LIBXSMM_EXPECT(EXIT_SUCCESS == libxsmm_dump("LIBXSMM-JIT-DUMP", name, code_ptr,
              /* dump executable code without constant data (apply_size vs info_size) */
              apply_size, 1/*unique*/, 0/*overwrite*/));
          }
#if defined(LIBXSMM_VTUNE)
          if (iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
            LIBXSMM_VTUNE_JIT_DESC_TYPE vtune_jit_desc;
            const unsigned int code_id = iJIT_GetNewMethodID();
            internal_get_vtune_jitdesc(code_ptr, code_id, apply_size, name, &vtune_jit_desc);
            iJIT_NotifyEvent(LIBXSMM_VTUNE_JIT_LOAD, &vtune_jit_desc);
            info->code_id = code_id;
          }
          else {
            info->code_id = 0;
          }
#endif
#if defined(LIBXSMM_PERF)
          /* If JIT is enabled and a valid name is given, emit information for profiler
           * In jitdump case this needs to be done after mprotect as it gets overwritten
           * otherwise. */
          libxsmm_perf_dump_code(code_ptr, apply_size, name);
#endif
        }
        if (NULL != info->reloc && info->pointer != info->reloc) {
#if defined(_WIN32)
          /* TODO: implement memory protection under Microsoft Windows */
#else
          /* memory is already protected at this point; relocate code */
          LIBXSMM_ASSERT(0 != (LIBXSMM_MALLOC_FLAG_MMAP & flags));
          *memory = code_ptr; /* relocate */
          info->pointer = info->reloc;
          info->reloc = NULL;
# if !defined(LIBXSMM_MALLOC_CRC_OFF) /* update checksum */
#   if defined(LIBXSMM_MALLOC_CRC_LIGHT)
          { const internal_malloc_info_type *const code_info = internal_malloc_info(code_ptr, 0/*no check*/);
            info->hash = LIBXSMM_CRCPTR(LIBXSMM_MALLOC_SEED, code_info);
          }
#   else
          info->hash = libxsmm_crc32(LIBXSMM_MALLOC_SEED, info,
            /* info size minus actual hash value */
            (unsigned int)(((char*)&info->hash) - ((char*)info)));
#   endif
# endif   /* treat memory protection errors as soft error; ignore return value */
          munmap(buffer, alloc_size);
# if defined(__APPLE__) && defined(__arm64__)
          if (0 == (LIBXSMM_MALLOC_FLAG_W & flags)) {
            pthread_jit_write_protect_np(1/*true*/);
          }
# endif
#endif
        }
#if !defined(_WIN32)
        else { /* malloc-based fallback */
# if !defined(LIBXSMM_MALLOC_CRC_OFF) && defined(LIBXSMM_VTUNE) /* check checksum */
#   if defined(LIBXSMM_MALLOC_CRC_LIGHT)
          assert(info->hash == LIBXSMM_CRCPTR(LIBXSMM_MALLOC_SEED, info)); /* !LIBXSMM_ASSERT */
#   else
          assert(info->hash == libxsmm_crc32(LIBXSMM_MALLOC_SEED, info, /* !LIBXSMM_ASSERT */
            /* info size minus actual hash value */
            (unsigned int)(((char*)&info->hash) - ((char*)info))));
#   endif
# endif
          result = libxsmm_malloc_xattrib(buffer, flags, alloc_size);
        }
#endif
      }
    }
  }
  else if (NULL == memory || NULL == *memory) {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_malloc_attrib failed because NULL cannot be attributed!\n");
    }
    result = EXIT_FAILURE;
  }
  else if ((LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity)
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM WARNING: %s buffer %p does not match!\n",
      0 != (LIBXSMM_MALLOC_FLAG_X & flags) ? "executable" : "memory", *memory);
  }
  return result;
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_MALLOC void* libxsmm_aligned_malloc(size_t size, size_t alignment)
{
  void* result = NULL;
  LIBXSMM_INIT
  if (2 > internal_malloc_kind) {
#if !defined(NDEBUG)
    int status =
#endif
    libxsmm_xmalloc(&result, size, alignment, LIBXSMM_MALLOC_FLAG_DEFAULT, NULL/*extra*/, 0/*extra_size*/);
    assert(EXIT_SUCCESS == status || NULL == result); /* !LIBXSMM_ASSERT */
  }
  else { /* scratch */
    const void *const caller = libxsmm_trace_caller_id(0/*level*/);
    internal_scratch_malloc(&result, size, alignment, LIBXSMM_MALLOC_FLAG_DEFAULT, caller);
  }
  return result;
}


LIBXSMM_API void* libxsmm_realloc(size_t size, void* ptr)
{
  const int nzeros = LIBXSMM_INTRINSICS_BITSCANFWD64((uintptr_t)ptr), alignment = 1 << nzeros;
  LIBXSMM_ASSERT(0 == ((uintptr_t)ptr & ~(0xFFFFFFFFFFFFFFFF << nzeros)));
  LIBXSMM_INIT
  if (2 > internal_malloc_kind) {
#if !defined(NDEBUG)
    int status =
#endif
    libxsmm_xmalloc(&ptr, size, alignment, LIBXSMM_MALLOC_FLAG_REALLOC, NULL/*extra*/, 0/*extra_size*/);
    assert(EXIT_SUCCESS == status || NULL == ptr); /* !LIBXSMM_ASSERT */
  }
  else { /* scratch */
    const void *const caller = libxsmm_trace_caller_id(0/*level*/);
    internal_scratch_malloc(&ptr, size, alignment, LIBXSMM_MALLOC_FLAG_REALLOC, caller);
  }
  return ptr;
}


LIBXSMM_API void* libxsmm_scratch_malloc(size_t size, size_t alignment, const void* caller)
{
  void* result;
  LIBXSMM_INIT
  internal_scratch_malloc(&result, size, alignment,
    LIBXSMM_MALLOC_INTERNAL_CALLER != caller ? LIBXSMM_MALLOC_FLAG_DEFAULT : LIBXSMM_MALLOC_FLAG_PRIVATE,
    caller);
  return result;
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_MALLOC void* libxsmm_malloc(size_t size)
{
  return libxsmm_aligned_malloc(size, 0/*auto*/);
}


LIBXSMM_API void libxsmm_free(const void* memory)
{
  if (NULL != memory) {
#if defined(LIBXSMM_MALLOC_SCRATCH_DELETE_FIRST) || /* prefer safe method if possible */ \
  !defined(LIBXSMM_MALLOC_HOOK)
# if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
    internal_malloc_pool_type *const pool = internal_scratch_malloc_pool(memory);
    if (NULL != pool) { /* memory belongs to scratch domain */
      internal_scratch_free(memory, pool);
    }
    else
# endif
    { /* local */
      libxsmm_xfree(memory, 2/*check*/);
    }
#else /* lookup matching pool */
    internal_malloc_info_type *const info = internal_malloc_info(memory, 2/*check*/);
    static int error_once = 0;
    if (NULL != info && 0 == (LIBXSMM_MALLOC_FLAG_SCRATCH & info->flags)) { /* !libxsmm_free */
# if !defined(NDEBUG)
      if (EXIT_SUCCESS != internal_xfree(memory, info) /* invalidates info */
        && 0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: memory deallocation failed!\n");
      }
# else
      internal_xfree(memory, info); /* !libxsmm_free, invalidates info */
# endif
    }
    else {
# if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
      internal_malloc_pool_type *const pool = internal_scratch_malloc_pool(memory);
      if (NULL != pool) { /* memory belongs to scratch domain */
        internal_scratch_free(memory, pool);
      }
      else
# endif
      {
# if defined(NDEBUG) && defined(LIBXSMM_MALLOC_HOOK)
        __real_free((void*)memory);
# else
#   if defined(LIBXSMM_MALLOC_HOOK)
        __real_free((void*)memory);
#   endif
        if (0 != libxsmm_verbosity && /* library code is expected to be mute */
            1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: deallocation does not match allocation!\n");
        }
# endif
      }
    }
#endif
  }
}


LIBXSMM_API_INTERN void libxsmm_xrelease_scratch(LIBXSMM_LOCK_TYPE(LIBXSMM_LOCK)* lock)
{
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  internal_malloc_pool_type* pools = NULL;
  libxsmm_scratch_info scratch_info;
  LIBXSMM_ASSERT(libxsmm_scratch_pools <= LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS);
  if (NULL != lock) {
    LIBXSMM_LOCK_ACQUIRE(LIBXSMM_LOCK, lock);
  }
# if defined(LIBXSMM_MALLOC_DELETE_SAFE)
  if (0 == (internal_malloc_kind & 1) || 0 >= internal_malloc_kind)
# endif
  {
    unsigned int i;
    pools = (internal_malloc_pool_type*)LIBXSMM_UP2(
      (uintptr_t)internal_malloc_pool_buffer, LIBXSMM_MALLOC_SCRATCH_PADDING);
    for (i = 0; i < libxsmm_scratch_pools; ++i) {
      if (0 != pools[i].instance.minsize) {
        if (
# if !defined(LIBXSMM_MALLOC_SCRATCH_DELETE_FIRST)
          1 < /*LIBXSMM_ATOMIC_LOAD(&*/pools[i].instance.counter/*, LIBXSMM_ATOMIC_SEQ_CST)*/ &&
# endif
          NULL != pools[i].instance.buffer)
        {
          internal_malloc_info_type *const info = internal_malloc_info(pools[i].instance.buffer, 2/*check*/);
          if (NULL != info) internal_xfree(info->pointer, info); /* invalidates info */
        }
      }
      else break; /* early exit */
    }
  }
  LIBXSMM_EXPECT(EXIT_SUCCESS == libxsmm_get_scratch_info(&scratch_info));
  if (0 != scratch_info.npending && /* library code is expected to be mute */
    (LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity))
  {
    char pending_size_buffer[32];
    libxsmm_format_value(pending_size_buffer, sizeof(pending_size_buffer),
      internal_malloc_public_cur + internal_malloc_local_cur, "KM", "B", 10);
    fprintf(stderr, "LIBXSMM WARNING: %s pending scratch-memory from %" PRIuPTR " allocation%s!\n",
      pending_size_buffer, (uintptr_t)scratch_info.npending, 1 < scratch_info.npending ? "s" : "");
  }
  if (NULL != pools) {
    memset(pools, 0, (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) * sizeof(internal_malloc_pool_type));
    /* no reset: keep private watermark (internal_malloc_private_max, internal_malloc_private_cur) */
    internal_malloc_public_max = internal_malloc_public_cur = 0;
    internal_malloc_local_max = internal_malloc_local_cur = 0;
    internal_malloc_scratch_nmallocs = 0;
  }
  if (NULL != lock) {
    LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK, lock);
  }
#endif
}


LIBXSMM_API void libxsmm_release_scratch(void)
{
  libxsmm_xrelease_scratch(&libxsmm_lock_global);
}


LIBXSMM_API int libxsmm_get_malloc_info(const void* memory, libxsmm_malloc_info* info)
{
  int result = EXIT_SUCCESS;
  if (NULL != info) {
    size_t size;
    result = libxsmm_get_malloc_xinfo(memory, &size, NULL/*flags*/, NULL/*extra*/);
    LIBXSMM_MEMZERO127(info);
    if (EXIT_SUCCESS == result) {
      info->size = size;
    }
#if !defined(NDEBUG) /* library code is expected to be mute */
    else if (LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity) {
      static int error_once = 0;
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
        fprintf(stderr, "LIBXSMM WARNING: foreign memory buffer %p discovered!\n", memory);
      }
    }
#endif
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API int libxsmm_get_scratch_info(libxsmm_scratch_info* info)
{
  int result = EXIT_SUCCESS;
  if (NULL != info) {
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
    LIBXSMM_MEMZERO127(info);
    info->nmallocs = internal_malloc_scratch_nmallocs;
    info->internal = internal_malloc_private_max;
    info->local = internal_malloc_local_max;
    info->size = internal_malloc_public_max;
    { const internal_malloc_pool_type* pool = (const internal_malloc_pool_type*)LIBXSMM_UP2(
        (uintptr_t)internal_malloc_pool_buffer, LIBXSMM_MALLOC_SCRATCH_PADDING);
# if (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
      const internal_malloc_pool_type *const end = pool + libxsmm_scratch_pools;
      LIBXSMM_ASSERT(libxsmm_scratch_pools <= LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS);
      for (; pool != end; ++pool) if ((LIBXSMM_MALLOC_INTERNAL_CALLER) != pool->instance.site) {
# endif
        if (0 != pool->instance.minsize) {
          const size_t npending = /*LIBXSMM_ATOMIC_LOAD(&*/pool->instance.counter/*, LIBXSMM_ATOMIC_RELAXED)*/;
# if defined(LIBXSMM_MALLOC_SCRATCH_DELETE_FIRST)
          info->npending += npending;
# else
          info->npending += 1 < npending ? (npending - 1) : 0;
# endif
          ++info->npools;
        }
# if (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
        else break; /* early exit */
      }
# endif
    }
#else
    LIBXSMM_MEMZERO127(info);
#endif /*defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))*/
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API void libxsmm_set_scratch_limit(size_t nbytes)
{
  /* !LIBXSMM_INIT */
  internal_malloc_scratch_limit = nbytes;
}


LIBXSMM_API size_t libxsmm_get_scratch_limit(void)
{
  size_t result;
  /* !LIBXSMM_INIT */
  if (LIBXSMM_SCRATCH_DEFAULT != internal_malloc_scratch_limit) {
    result = internal_malloc_scratch_limit;
  }
  else if (0 == internal_malloc_kind) {
    result = LIBXSMM_MALLOC_SCRATCH_LIMIT;
  }
  else {
    result = LIBXSMM_SCRATCH_UNLIMITED;
  }
  return result;
}


LIBXSMM_API void libxsmm_set_malloc(int enabled, const size_t* lo, const size_t* hi)
{
  /* !LIBXSMM_INIT */
#if defined(LIBXSMM_MALLOC_HOOK) && defined(LIBXSMM_MALLOC) && (0 != LIBXSMM_MALLOC)
# if (0 < LIBXSMM_MALLOC)
  LIBXSMM_UNUSED(enabled);
  internal_malloc_kind = LIBXSMM_MALLOC;
# else
  internal_malloc_kind = enabled;
# endif
  /* setup lo/hi after internal_malloc_kind! */
  if (NULL != lo) internal_malloc_limit[0] = *lo;
  if (NULL != hi) {
    const size_t scratch_limit = libxsmm_get_scratch_limit();
    const size_t malloc_upper = LIBXSMM_MIN(*hi, scratch_limit);
    internal_malloc_limit[1] = LIBXSMM_MAX(malloc_upper, internal_malloc_limit[0]);
  }
#else
  LIBXSMM_UNUSED(lo); LIBXSMM_UNUSED(hi);
  internal_malloc_kind = enabled;
#endif
  libxsmm_malloc_init();
}


LIBXSMM_API int libxsmm_get_malloc(size_t* lo, size_t* hi)
{
  LIBXSMM_INIT
#if defined(LIBXSMM_MALLOC_HOOK) && defined(LIBXSMM_MALLOC) && (0 != LIBXSMM_MALLOC)
  if (NULL != lo) *lo = internal_malloc_limit[0];
  if (NULL != hi) *hi = internal_malloc_limit[1];
#else
  if (NULL != lo) *lo = 0;
  if (NULL != hi) *hi = 0;
#endif
  return internal_malloc_kind;
}


LIBXSMM_API void libxsmm_pmalloc_init(size_t size, size_t* num, void* pool[], void* storage)
{
  char* p = (char*)storage;
  volatile int* lock;
  size_t n, i = 0;
  LIBXSMM_ASSERT(0 < size && NULL != num && NULL != pool && NULL != storage);
  LIBXSMM_INIT /* CRC-facility must be initialized upfront */
  lock = internal_pmallocs + LIBXSMM_MOD2(LIBXSMM_CRCPTR(LIBXSMM_MALLOC_SEED, pool), LIBXSMM_MALLOC_NLOCKS);
  LIBXSMM_ATOMIC_ACQUIRE(lock, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_SEQ_CST);
  for (n = *num; i < n; ++i, p += size) pool[i] = p;
  LIBXSMM_ATOMIC_RELEASE(lock, LIBXSMM_ATOMIC_SEQ_CST);
}


LIBXSMM_API void* libxsmm_pmalloc(void* pool[], size_t* i)
{
  const unsigned int hash = LIBXSMM_CRCPTR(LIBXSMM_MALLOC_SEED, pool);
  volatile int *const lock = internal_pmallocs + LIBXSMM_MOD2(hash, LIBXSMM_MALLOC_NLOCKS);
  void* pointer;
  LIBXSMM_ASSERT(NULL != pool && NULL != i);
  LIBXSMM_ATOMIC_ACQUIRE(lock, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_SEQ_CST);
  assert(0 < *i && ((size_t)-1) != *i); /* !LIBXSMM_ASSERT */
  pointer = pool[--(*i)];
#if !defined(NDEBUG)
  pool[*i] = NULL;
#endif
  LIBXSMM_ATOMIC_RELEASE(lock, LIBXSMM_ATOMIC_SEQ_CST);
  LIBXSMM_ASSERT(NULL != pointer);
  return pointer;
}


LIBXSMM_API void libxsmm_pfree(void* pointer, void* pool[], size_t* i)
{
  const unsigned int hash = LIBXSMM_CRCPTR(LIBXSMM_MALLOC_SEED, pool);
  volatile int *const lock = internal_pmallocs + LIBXSMM_MOD2(hash, LIBXSMM_MALLOC_NLOCKS);
  LIBXSMM_ASSERT(NULL != pointer && NULL != pool && NULL != i);
  LIBXSMM_ATOMIC_ACQUIRE(lock, LIBXSMM_SYNC_NPAUSE, LIBXSMM_ATOMIC_SEQ_CST);
  assert(NULL == pool[*i]); /* !LIBXSMM_ASSERT */
  pool[(*i)++] = pointer;
  LIBXSMM_ATOMIC_RELEASE(lock, LIBXSMM_ATOMIC_SEQ_CST);
}
