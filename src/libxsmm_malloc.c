/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
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

/* must be defined *before* other files are included */
#if !defined(_GNU_SOURCE)
# define _GNU_SOURCE
#endif
#include <libxsmm.h>
#include "libxsmm_trace.h"
#include "libxsmm_main.h"
#include "libxsmm_hash.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(__TBB)
# include <tbb/scalable_allocator.h>
#endif
#if defined(_WIN32)
# include <windows.h>
#else
# include <sys/mman.h>
# if defined(MAP_HUGETLB) && defined(MAP_POPULATE)
#   include <sys/utsname.h>
#   include <string.h>
# endif
# include <sys/types.h>
# include <unistd.h>
# include <errno.h>
# if defined(MAP_ANONYMOUS)
#   define LIBXSMM_MAP_ANONYMOUS MAP_ANONYMOUS
# else
#   define LIBXSMM_MAP_ANONYMOUS MAP_ANON
# endif
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
typedef struct LineNumberInfo {
  unsigned int Offset;
  unsigned int LineNumber;
} LineNumberInfo;
typedef struct iJIT_Method_Load_V2 {
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
#   define LIBXSMM_MALLOC_FALLBACK 4
# endif
#else
# if !defined(LIBXSMM_MALLOC_FALLBACK)
#   define LIBXSMM_MALLOC_FALLBACK 0
# endif
#endif /*defined(LIBXSMM_VTUNE)*/
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif
#if defined(LIBXSMM_PERF)
# include "libxsmm_perf.h"
#endif

#if !defined(LIBXSMM_MALLOC_NOCRC)
# if defined(NDEBUG)
#   define LIBXSMM_MALLOC_NOCRC
# elif !defined(LIBXSMM_BUILD)
#   define LIBXSMM_MALLOC_NOCRC
# endif
# if !defined(LIBXSMM_MALLOC_NOCRC) && !defined(LIBXSMM_MALLOC_SEED)
#   define LIBXSMM_MALLOC_SEED 1051981
# endif
#endif

#if !defined(LIBXSMM_MALLOC_ALIGNMAX)
# define LIBXSMM_MALLOC_ALIGNMAX (2 * 1024 *1024)
#endif
#if !defined(LIBXSMM_MALLOC_ALIGNFCT)
# define LIBXSMM_MALLOC_ALIGNFCT 8
#endif

#if !defined(LIBXSMM_MALLOC_SCRATCH_XFREE)
# define LIBXSMM_MALLOC_SCRATCH_XFREE
#endif

/* perform low-level allocation even for small non-executable buffers */
#if !defined(LIBXSMM_MALLOC_MMAP)
/*# define LIBXSMM_MALLOC_MMAP*/
#endif


typedef struct LIBXSMM_RETARGETABLE internal_malloc_info_type {
  libxsmm_free_function free;
  void *context, *pointer, *reloc;
  size_t size;
  int flags;
#if defined(LIBXSMM_VTUNE)
  unsigned int code_id;
#endif
#if !defined(LIBXSMM_MALLOC_NOCRC) /* hash *must* be the last entry */
  unsigned int hash;
#endif
} internal_malloc_info_type;

typedef struct LIBXSMM_RETARGETABLE internal_malloc_pool_type {
  char *buffer, *head;
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  const void* site;
#endif
  size_t minsize;
  size_t counter;
} internal_malloc_pool_type;

/** Scratch pool, which supports up to MAX_NSCRATCH allocation sites. */
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
LIBXSMM_API_VARIABLE internal_malloc_pool_type internal_malloc_scratch_pool[LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS];
#endif
LIBXSMM_API_VARIABLE size_t internal_malloc_scratch_nmallocs;


LIBXSMM_API_DEFINITION size_t libxsmm_gcd(size_t a, size_t b)
{
  while (0 != b) {
    const size_t r = a % b;
    a = b;
    b = r;
  }
  return a;
}


LIBXSMM_API_DEFINITION size_t libxsmm_lcm(size_t a, size_t b)
{
  return (a * b) / libxsmm_gcd(a, b);
}


LIBXSMM_API_DEFINITION size_t libxsmm_alignment(size_t size, size_t alignment)
{
  size_t result = sizeof(void*);
  if ((LIBXSMM_MALLOC_ALIGNFCT * LIBXSMM_MALLOC_ALIGNMAX) <= size) {
    result = libxsmm_lcm(0 == alignment ? (LIBXSMM_ALIGNMENT) : libxsmm_lcm(alignment, LIBXSMM_ALIGNMENT), LIBXSMM_MALLOC_ALIGNMAX);
  }
  else {
    if ((LIBXSMM_MALLOC_ALIGNFCT * LIBXSMM_ALIGNMENT) <= size) {
      result = (0 == alignment ? (LIBXSMM_ALIGNMENT) : libxsmm_lcm(alignment, LIBXSMM_ALIGNMENT));
    }
    else if (0 != alignment) {
      result = libxsmm_lcm(alignment, result);
    }
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_xset_default_allocator(LIBXSMM_LOCK_TYPE* lock,
  void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn)
{
  int result = EXIT_SUCCESS;
  if (0 != lock) {
    LIBXSMM_INIT
    LIBXSMM_LOCK_ACQUIRE(lock);
  }
  if (0 != malloc_fn.function && 0 != free_fn.function) {
    libxsmm_default_allocator_context = context;
    libxsmm_default_malloc_fn = malloc_fn;
    libxsmm_default_free_fn = free_fn;
  }
  else {
    void* internal_allocator = 0;
    libxsmm_malloc_function internal_malloc_fn;
    libxsmm_free_function internal_free_fn;
#if defined(__TBB)
    internal_allocator = 0;
    internal_malloc_fn.function = scalable_malloc;
    internal_free_fn.function = scalable_free;
#else
    internal_allocator = 0;
    internal_malloc_fn.function = malloc;
    internal_free_fn.function = free;
#endif
    if (0 == malloc_fn.function && 0 == free_fn.function) {
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
      if (0 == libxsmm_default_malloc_fn.function || 0 == libxsmm_default_free_fn.function) {
        libxsmm_default_allocator_context = internal_allocator;
        libxsmm_default_malloc_fn = internal_malloc_fn;
        libxsmm_default_free_fn = internal_free_fn;
      }
      result = EXIT_FAILURE;
    }
  }
  if (0 != lock) {
    LIBXSMM_LOCK_RELEASE(lock);
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_xget_default_allocator(LIBXSMM_LOCK_TYPE* lock,
  void** context, libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn)
{
  int result = EXIT_SUCCESS;
  if (0 != context || 0 != malloc_fn || 0 != free_fn) {
    if (0 != lock) {
      LIBXSMM_INIT
      LIBXSMM_LOCK_ACQUIRE(lock);
    }
    if (context) *context = libxsmm_default_allocator_context;
    if (0 != malloc_fn) *malloc_fn = libxsmm_default_malloc_fn;
    if (0 != free_fn) *free_fn = libxsmm_default_free_fn;
    if (0 != lock) {
      LIBXSMM_LOCK_RELEASE(lock);
    }
  }
  else if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM ERROR: invalid signature used to get the default memory allocator!\n");
    }
    result = EXIT_FAILURE;
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_xset_scratch_allocator(LIBXSMM_LOCK_TYPE* lock,
  void* context, libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn)
{
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (0 != lock) {
    LIBXSMM_INIT
    LIBXSMM_LOCK_ACQUIRE(lock);
  }
  /* make sure the default allocator is setup before adopting it eventually */
  if (0 == libxsmm_default_malloc_fn.function || 0 == libxsmm_default_free_fn.function) {
    const libxsmm_malloc_function null_malloc_fn = { 0 };
    const libxsmm_free_function null_free_fn = { 0 };
    libxsmm_xset_default_allocator(lock, 0/*context*/, null_malloc_fn, null_free_fn);
  }
  if (0 == malloc_fn.function && 0 == free_fn.function) { /* adopt default allocator */
    libxsmm_scratch_allocator_context = libxsmm_default_allocator_context;
    libxsmm_scratch_malloc_fn = libxsmm_default_malloc_fn;
    libxsmm_scratch_free_fn = libxsmm_default_free_fn;
  }
  else if (0 != malloc_fn.function) {
    if (0 == free_fn.function
      && /*warning*/(1 < libxsmm_verbosity || 0 > libxsmm_verbosity)
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
    if (0 == libxsmm_scratch_malloc_fn.function) {
      libxsmm_scratch_allocator_context = libxsmm_default_allocator_context;
      libxsmm_scratch_malloc_fn = libxsmm_default_malloc_fn;
      libxsmm_scratch_free_fn = libxsmm_default_free_fn;
    }
    result = EXIT_FAILURE;
  }
  if (0 != lock) {
    LIBXSMM_LOCK_RELEASE(lock);
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_xget_scratch_allocator(LIBXSMM_LOCK_TYPE* lock,
  void** context, libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn)
{
  int result = EXIT_SUCCESS;
  if (0 != context || 0 != malloc_fn || 0 != free_fn) {
    if (0 != lock) {
      LIBXSMM_INIT
      LIBXSMM_LOCK_ACQUIRE(lock);
    }
    if (context) *context = libxsmm_scratch_allocator_context;
    if (0 != malloc_fn) *malloc_fn = libxsmm_scratch_malloc_fn;
    if (0 != free_fn) *free_fn = libxsmm_scratch_free_fn;
    if (0 != lock) {
      LIBXSMM_LOCK_RELEASE(lock);
    }
  }
  else if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM ERROR: invalid signature used to get the scratch memory allocator!\n");
    }
    result = EXIT_FAILURE;
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_set_default_allocator(void* context,
  libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn)
{
  return libxsmm_xset_default_allocator(&libxsmm_lock_global, context, malloc_fn, free_fn);
}


LIBXSMM_API_DEFINITION int libxsmm_get_default_allocator(void** context,
  libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn)
{
  return libxsmm_xget_default_allocator(&libxsmm_lock_global, context, malloc_fn, free_fn);
}


LIBXSMM_API_DEFINITION int libxsmm_set_scratch_allocator(void* context,
  libxsmm_malloc_function malloc_fn, libxsmm_free_function free_fn)
{
  return libxsmm_xset_scratch_allocator(&libxsmm_lock_global, context, malloc_fn, free_fn);
}


LIBXSMM_API_DEFINITION int libxsmm_get_scratch_allocator(void** context,
  libxsmm_malloc_function* malloc_fn, libxsmm_free_function* free_fn)
{
  return libxsmm_xget_scratch_allocator(&libxsmm_lock_global, context, malloc_fn, free_fn);
}


LIBXSMM_API_INLINE internal_malloc_info_type* internal_malloc_info(const void* memory)
{
  internal_malloc_info_type *const result = (internal_malloc_info_type*)
    (0 != memory ? (((const char*)memory) - sizeof(internal_malloc_info_type)) : 0);
#if defined(LIBXSMM_MALLOC_NOCRC)
  return result;
#else /* calculate checksum over info */
  return (0 != result && result->hash == libxsmm_crc32(result, /* info size minus actual hash value */
    (unsigned int)(((char*)&result->hash) - ((char*)result)), LIBXSMM_MALLOC_SEED)) ? result : 0;
#endif
}


LIBXSMM_API_DEFINITION int libxsmm_get_malloc_xinfo(const void* memory, size_t* size, int* flags, void** extra)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG) || !defined(LIBXSMM_MALLOC_NOCRC)
  static int error_once = 0;
  if (0 != size || 0 != extra)
#endif
  {
    const internal_malloc_info_type *const info = internal_malloc_info(memory);
    if (0 != info) {
      if (size) *size = info->size;
      if (flags) *flags = info->flags;
      if (extra) *extra = info->pointer;
    }
    else {
      if (0 != memory) {
#if !defined(LIBXSMM_MALLOC_NOCRC)
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
         && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: checksum error for memory buffer %p!\n", memory);
        }
#endif
        result = EXIT_FAILURE;
      }
      if (size) *size = 0;
      if (flags) *flags = 0;
      if (extra) *extra = 0;
    }
  }
#if !defined(NDEBUG)
  else {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: attachment error for memory buffer %p!\n", memory);
    }
    result = EXIT_FAILURE;
  }
  assert(EXIT_SUCCESS == result);
#endif
  return result;
}


#if !defined(_WIN32)

LIBXSMM_API_INLINE void internal_mhint(void* buffer, size_t size)
{
  assert((MAP_FAILED != buffer && 0 != buffer) || 0 == size);
  /* proceed after failed madvise (even in case of an error; take what we got) */
  /* issue no warning as a failure seems to be related to the kernel version */
  madvise(buffer, size, MADV_NORMAL/*MADV_RANDOM*/
#if defined(MADV_NOHUGEPAGE) /* if not available, we then take what we got (THP) */
    | ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) > size ? MADV_NOHUGEPAGE : 0)
#endif
#if defined(MADV_DONTDUMP)
    | ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) > size ? 0 : MADV_DONTDUMP)
#endif
  );
}


LIBXSMM_API_INLINE void* internal_xmap(const char* dir, size_t size, int flags, void** rx)
{
  void* result = MAP_FAILED;
  char filename[4096];
  int i = LIBXSMM_SNPRINTF(filename, sizeof(filename), "%s/.libxsmm_XXXXXX.jit", dir);
  assert(0 != rx);
  if (0 <= i && i < (int)sizeof(filename)) {
    i = mkstemps(filename, 4);
    if (-1 != i && 0 == unlink(filename) && 0 == ftruncate(i, size)) {
      void *const xmap = mmap(0, size, PROT_READ | PROT_EXEC, flags | MAP_SHARED /*| LIBXSMM_MAP_ANONYMOUS*/, i, 0);
      if (MAP_FAILED != xmap) {
        assert(0 != xmap);
        result = mmap(0, size, PROT_READ | PROT_WRITE, flags | MAP_SHARED /*| LIBXSMM_MAP_ANONYMOUS*/, i, 0);
        if (MAP_FAILED != result) {
          assert(0 != result);
          internal_mhint(xmap, size);
          *rx = xmap;
        }
        else {
          munmap(xmap, size);
        }
      }
    }
  }
  return result;
}

#endif /*!defined(_WIN32)*/


LIBXSMM_API_DEFINITION int libxsmm_xmalloc(void** memory, size_t size, size_t alignment,
  int flags, const void* extra, size_t extra_size)
{
  int result = EXIT_SUCCESS;
  if (memory) {
    static int error_once = 0;
    if (0 < size) {
      const size_t internal_size = size + extra_size + sizeof(internal_malloc_info_type);
      /* ATOMIC BEGIN: this region should be atomic/locked */
        void* context = libxsmm_default_allocator_context;
        libxsmm_malloc_function malloc_fn = libxsmm_default_malloc_fn;
        libxsmm_free_function free_fn = libxsmm_default_free_fn;
      /* ATOMIC END: this region should be atomic */
      size_t alloc_alignment = 0, alloc_size = 0;
      void *alloc_failed = 0, *buffer = 0, *reloc = 0;
      if (0 != (LIBXSMM_MALLOC_FLAG_SCRATCH & flags)) {
        context = libxsmm_scratch_allocator_context;
        malloc_fn = libxsmm_scratch_malloc_fn;
        free_fn = libxsmm_scratch_free_fn;
      }
      flags |= LIBXSMM_MALLOC_FLAG_RW; /* normalize given flags since flags=0 is accepted as well */
#if !defined(LIBXSMM_MALLOC_MMAP)
      if (0 == (LIBXSMM_MALLOC_FLAG_X & flags) && 0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) {
        alloc_alignment = (0 == alignment ? libxsmm_alignment(size, alignment) : alignment);
        alloc_size = internal_size + alloc_alignment - 1;
        buffer = 0 != malloc_fn.function
          ? (0 == context ? malloc_fn.function(alloc_size) : malloc_fn.ctx_form(context, alloc_size))
          : 0;
      }
      else
#endif
      {
#if defined(_WIN32)
        const int xflags = (0 != (LIBXSMM_MALLOC_FLAG_X & flags) ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
        if ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) > size) {
          alloc_alignment = (0 == alignment ? libxsmm_alignment(size, alignment) : alignment);
          alloc_size = internal_size + alloc_alignment - 1;
          buffer = VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, xflags);
        }
        else {
          HANDLE process_token;
          const SIZE_T alloc_alignmax = GetLargePageMinimum();
          /* respect user-requested alignment */
          alloc_alignment = 0 == alignment ? alloc_alignmax : libxsmm_lcm(alignment, alloc_alignmax);
          alloc_size = LIBXSMM_UP2(internal_size, alloc_alignment); /* assume that alloc_alignment is POT */
          if (TRUE == OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &process_token)) {
            TOKEN_PRIVILEGES tp;
            if (TRUE == LookupPrivilegeValue(NULL, TEXT("SeLockMemoryPrivilege"), &tp.Privileges[0].Luid)) {
              tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED; tp.PrivilegeCount = 1; /* enable privilege */
              if ( TRUE == AdjustTokenPrivileges(process_token, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0)
                && ERROR_SUCCESS == GetLastError()/*may has failed (regardless of TRUE)*/)
              {
                buffer = VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, xflags);
              }
              tp.Privileges[0].Attributes = 0; /* disable privilege */
              AdjustTokenPrivileges(process_token, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
            }
            CloseHandle(process_token);
          }
          if (alloc_failed == buffer) { /* retry allocation with regular page size */
            alloc_alignment = (0 == alignment ? libxsmm_alignment(size, alignment) : alignment);
            alloc_size = internal_size + alloc_alignment - 1;
            buffer = VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, xflags);
          }
        }
        if (alloc_failed != buffer) {
          flags |= LIBXSMM_MALLOC_FLAG_MMAP; /* select the corresponding deallocation */
        }
        else if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) { /* fall-back allocation */
          buffer = 0 != malloc_fn.function
            ? (0 == context ? malloc_fn.function(alloc_size) : malloc_fn.ctx_form(context, alloc_size))
            : 0;
        }
#else /* !defined(_WIN32) */
        int xflags = 0
# if defined(MAP_NORESERVE)
          | ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) > size ? MAP_NORESERVE : 0)
# endif
# if defined(MAP_32BIT)
          | ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) > size ? MAP_32BIT : 0)
# endif
# if defined(MAP_HUGETLB) /* may fail depending on system settings */
          | ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) > size ? 0 : MAP_HUGETLB)
# endif
# if defined(MAP_UNINITIALIZED) /* unlikely to be available */
          | MAP_UNINITIALIZED
# endif
# if defined(MAP_LOCKED) && /*disadvantage*/0
          | MAP_LOCKED
# endif
        ;
        /* prefault pages to avoid data race in Linux' page-fault handler pre-3.10.0-327 */
# if defined(MAP_HUGETLB) && defined(MAP_POPULATE)
        struct utsname osinfo;
        if (0 != (MAP_HUGETLB & xflags) && 0 <= uname(&osinfo) && 0 == strcmp("Linux", osinfo.sysname)) {
          unsigned int version_major = 3, version_minor = 10, version_update = 0, version_patch = 327;
          if (4 == sscanf(osinfo.release, "%u.%u.%u-%u", &version_major, &version_minor, &version_update, &version_patch) &&
            LIBXSMM_VERSION4(3, 10, 0, 327) > LIBXSMM_VERSION4(version_major, version_minor, version_update, version_patch))
          {
            /* TODO: lock across threads and processes */
            xflags |= MAP_POPULATE;
          }
        }
# endif
        alloc_alignment = (0 == alignment ? libxsmm_alignment(size, alignment) : alignment);
        alloc_size = internal_size + alloc_alignment - 1;
        alloc_failed = MAP_FAILED;
        if (0 == (LIBXSMM_MALLOC_FLAG_X & flags)) {
          buffer = mmap(0, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | LIBXSMM_MAP_ANONYMOUS | xflags, -1, 0);
        }
        else {
          static LIBXSMM_TLS int fallback = -1;
          if (0 > fallback) { /* initialize fall-back allocation method */
            const char *const env = getenv("LIBXSMM_SE");
            fallback = (0 == env || 0 == *env || 0 != atoi(env)) ? LIBXSMM_MALLOC_FALLBACK : 4;
          }
          if (0 == fallback) {
            buffer = internal_xmap("/tmp", alloc_size, xflags, &reloc);
            if (alloc_failed == buffer) fallback = 1;
          }
          if (1 <= fallback) { /* continue with fall-back */
            if (1 == fallback) { /* 2nd try */
              buffer = internal_xmap(".", alloc_size, xflags, &reloc);
              if (alloc_failed == buffer) fallback = 2;
            }
            if (2 <= fallback) { /* continue with fall-back */
              if (2 == fallback) { /* 3rd try */
                buffer = internal_xmap(getenv("HOME"), alloc_size, xflags, &reloc);
                if (alloc_failed == buffer) fallback = 3;
              }
              if (3 <= fallback) { /* continue with fall-back */
                if (3 == fallback) { /* 4th try */
                  buffer = internal_xmap(getenv("JITDUMPDIR"), alloc_size, xflags, &reloc);
                  if (alloc_failed == buffer) fallback = 4;
                }
                if (4 <= fallback) { /* continue with fall-back */
                  if (4 == fallback) { /* 5th try */
                    buffer = mmap(0, alloc_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | LIBXSMM_MAP_ANONYMOUS | xflags, -1, 0);
                    if (alloc_failed == buffer) fallback = 5;
                  }
                  if (5 == fallback && alloc_failed != buffer) { /* final */
                    buffer = alloc_failed; /* trigger fall-back */
                  }
                }
              }
            }
          }
        }
        if (alloc_failed != buffer) {
          assert(0 != buffer);
          flags |= LIBXSMM_MALLOC_FLAG_MMAP; /* select the corresponding deallocation */
        }
        else {
          if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) { /* fall-back allocation */
            buffer = 0 != malloc_fn.function
              ? (0 == context ? malloc_fn.function(alloc_size) : malloc_fn.ctx_form(context, alloc_size))
              : 0;
            reloc = buffer;
          }
          else {
            reloc = 0;
          }
        }
        if (MAP_FAILED != buffer && 0 != buffer) {
          internal_mhint(buffer, alloc_size);
        }
#endif
      }
      if (alloc_failed != buffer && /*fall-back*/0 != buffer) {
        char *const aligned = LIBXSMM_ALIGN(((char*)buffer) + extra_size + sizeof(internal_malloc_info_type), alloc_alignment);
        internal_malloc_info_type *const info = (internal_malloc_info_type*)(aligned - sizeof(internal_malloc_info_type));
        assert((aligned + size) <= (((char*)buffer) + alloc_size));
        if (0 != extra) memcpy(buffer, extra, extra_size);
#if !defined(NDEBUG)
        else if (0 == extra && 0 != extra_size) {
          result = EXIT_FAILURE;
        }
#endif
        if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) {
          info->context = context;
          info->free = free_fn;
        }
        else {
          info->free.function = 0;
          info->context = 0;
        }
        info->pointer = buffer;
        info->reloc = reloc;
        info->size = size;
        info->flags = flags;
#if !defined(LIBXSMM_MALLOC_NOCRC) /* calculate checksum over info */
        info->hash = libxsmm_crc32(info, /* info size minus actual hash value */
          (unsigned int)(((char*)&info->hash) - ((char*)info)), LIBXSMM_MALLOC_SEED);
#endif
        *memory = aligned;
      }
      else {
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
         && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: memory allocation error for size %llu with flags=%i!\n",
            (unsigned long long)alloc_size, flags);
        }
        result = EXIT_FAILURE;
      }
    }
    else {
      if ((1 < libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM WARNING: zero-sized memory allocation detected!\n");
      }
      *memory = 0;
    }
  }
  else if (0 != size) {
    result = EXIT_FAILURE;
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_xfree(const void* memory)
{
  /*const*/ internal_malloc_info_type *const info = internal_malloc_info(memory);
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (0 != info) {
    void *const buffer = info->pointer;
#if !defined(LIBXSMM_BUILD) /* sanity check */
    if (0 != buffer || 0 == info->size)
#endif
    {
      assert(0 != buffer || 0 == info->size);
      if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & info->flags)) {
        if (0 != info->free.function) {
          if (0 == info->context) {
            info->free.function(buffer);
          }
          else {
            info->free.ctx_form(info->context, buffer);
          }
        }
      }
      else {
#if defined(LIBXSMM_VTUNE)
        if (0 != (LIBXSMM_MALLOC_FLAG_X & info->flags) && 0 != info->code_id && iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
          iJIT_NotifyEvent(LIBXSMM_VTUNE_JIT_UNLOAD, &info->code_id);
        }
#endif
#if defined(_WIN32)
        result = (0 == buffer || FALSE != VirtualFree(buffer, 0, MEM_RELEASE)) ? EXIT_SUCCESS : EXIT_FAILURE;
#else /* defined(_WIN32) */
        {
          const size_t alloc_size = info->size + (((const char*)memory) - ((const char*)buffer));
          void *const reloc = info->reloc;
          const int flags = info->flags;
          if (0 != munmap(buffer, alloc_size)) {
            if (0 != libxsmm_verbosity /* library code is expected to be mute */
             && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
            {
              const char *const error_message = strerror(errno);
              fprintf(stderr, "LIBXSMM ERROR: %s (munmap error #%i for range %p+%llu)!\n",
                error_message, errno, buffer, (unsigned long long)alloc_size);
            }
            result = EXIT_FAILURE;
          }
          if (0 != (LIBXSMM_MALLOC_FLAG_X & flags) && EXIT_SUCCESS == result
           && 0 != reloc && MAP_FAILED != reloc && buffer != reloc
           && 0 != munmap(reloc, alloc_size))
          {
            if (0 != libxsmm_verbosity /* library code is expected to be mute */
             && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
            {
              const char *const error_message = strerror(errno);
              fprintf(stderr, "LIBXSMM ERROR: %s (munmap error #%i for range %p+%llu)!\n",
                error_message, errno, reloc, (unsigned long long)alloc_size);
            }
            result = EXIT_FAILURE;
          }
        }
#endif
      }
    }
#if !defined(LIBXSMM_BUILD)
    else if ((1 < libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM WARNING: attempt to release memory from non-matching implementation!\n");
    }
#endif
  }
  else if (0 != memory) {
#if !defined(LIBXSMM_MALLOC_NOCRC)
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: checksum error for memory buffer %p!\n", memory);
    }
#endif
    result = EXIT_FAILURE;
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


#if defined(LIBXSMM_VTUNE)
LIBXSMM_API_INLINE void internal_get_vtune_jitdesc(const void* code,
  unsigned int code_id, size_t code_size, const char* code_name,
  LIBXSMM_VTUNE_JIT_DESC_TYPE* desc)
{
  assert(0 != code && 0 != code_id && 0 != code_size && 0 != desc);
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


LIBXSMM_API_DEFINITION int libxsmm_malloc_attrib(void** memory, int flags, const char* name)
{
  internal_malloc_info_type *const info = 0 != memory ? internal_malloc_info(*memory) : 0;
  int result = EXIT_SUCCESS;
  static int error_once = 0;
  if (0 != info) {
    void *const buffer = info->pointer;
    const size_t size = info->size;
#if defined(_WIN32)
    assert(0 != buffer || 0 == size);
#else
    assert((0 != buffer && MAP_FAILED != buffer) || 0 == size);
#endif
    /* quietly keep the read permission, but eventually revoke write permissions */
    if (0 == (LIBXSMM_MALLOC_FLAG_W & flags) || 0 != (LIBXSMM_MALLOC_FLAG_X & flags)) {
      const size_t alignment = (size_t)(((const char*)(*memory)) - ((const char*)buffer));
      const size_t alloc_size = size + alignment;
      if (0 == (LIBXSMM_MALLOC_FLAG_X & flags)) {
#if defined(_WIN32)
        /* TODO: implement memory protection under Microsoft Windows */
        LIBXSMM_UNUSED(alloc_size);
#else
        /* treat memory protection errors as soft error; ignore return value */
        mprotect(buffer, alloc_size/*entire memory region*/, PROT_READ);
#endif
      }
      else {
        void *const code_ptr =
#if !defined(_WIN32)
          0 != (LIBXSMM_MALLOC_FLAG_MMAP & flags) ? ((void*)(((char*)info->reloc) + alignment)) :
#endif
          *memory;
        assert(0 != (LIBXSMM_MALLOC_FLAG_X & flags));
        if (name && *name) { /* profiler support requested */
          if (0 > libxsmm_verbosity) { /* avoid dump when only the profiler is enabled */
            FILE *const code_file = fopen(name, "wb");
            if (0 != code_file) { /* dump byte-code into a file and print function pointer and filename */
              fprintf(stderr, "LIBXSMM-JIT-DUMP(ptr:file) %p : %s\n", code_ptr, name);
              fwrite(code_ptr, 1, size, code_file);
              fclose(code_file);
            }
          }
#if defined(LIBXSMM_VTUNE)
          if (iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
            LIBXSMM_VTUNE_JIT_DESC_TYPE vtune_jit_desc;
            const unsigned int code_id = iJIT_GetNewMethodID();
            internal_get_vtune_jitdesc(code_ptr, code_id, size, name, &vtune_jit_desc);
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
          libxsmm_perf_dump_code(code_ptr, size, name);
#endif
        }
        if (0 != (LIBXSMM_MALLOC_FLAG_MMAP & flags)) {
#if defined(_WIN32)
          /* TODO: implement memory protection under Microsoft Windows */
#else
          /* memory is already protected at this point; relocate code */
          assert(info->pointer != info->reloc);
          *memory = code_ptr; /* relocate */
          info->pointer = info->reloc;
          info->reloc = 0;
# if !defined(LIBXSMM_MALLOC_NOCRC) /* update checksum */
          info->hash = libxsmm_crc32(info, /* info size minus actual hash value */
            (unsigned int)(((char*)&info->hash) - ((char*)info)), LIBXSMM_MALLOC_SEED);
# endif
          /* treat memory protection errors as soft error; ignore return value */
          munmap(buffer, alloc_size);
#endif
        }
#if !defined(_WIN32)
        else { /* malloc-based fall-back */
# if !defined(LIBXSMM_MALLOC_NOCRC) && defined(LIBXSMM_VTUNE) /* update checksum */
          info->hash = libxsmm_crc32(info, /* info size minus actual hash value */
            (unsigned int)(((char*)&info->hash) - ((char*)info)), LIBXSMM_MALLOC_SEED);
# endif
          /* treat memory protection errors as soft error; ignore return value */
          mprotect(buffer, alloc_size/*entire memory region*/, PROT_READ | PROT_EXEC);
        }
#endif
      }
    }
  }
  else if (0 == memory || 0 == *memory) {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_malloc_attrib failed because NULL cannot be attributed!\n");
    }
    result = EXIT_FAILURE;
  }
  else {
    assert(0 != memory && 0 != *memory);
#if !defined(LIBXSMM_MALLOC_NOCRC)
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
     && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: checksum error for %s buffer %p!\n",
        0 != (LIBXSMM_MALLOC_FLAG_X & flags) ? "executable" : "memory", *memory);
    }
#endif
    result = EXIT_FAILURE;
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION void* libxsmm_aligned_malloc(size_t size, size_t alignment)
{
  void* result = 0;
  LIBXSMM_INIT
  return 0 == libxsmm_xmalloc(&result, size, alignment, LIBXSMM_MALLOC_FLAG_DEFAULT,
    0/*extra*/, 0/*extra_size*/) ? result : 0;
}


LIBXSMM_API_INLINE unsigned int internal_malloc_site(unsigned int* npools, unsigned int* hit, const void** site)
{
  assert(0 != npools && 0 != hit && 0 != site);
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  *npools = LIBXSMM_MIN(libxsmm_scratch_pools, LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS);
  if (1 < *npools) {
    if (0 != *site) {
      unsigned int i;
      for (i = 0; i < *npools; ++i) {
        if (*site == internal_malloc_scratch_pool[i].site) {
          *hit = 1; return i;
        }
      }
      *hit = 0;
    }
    else {
#if defined(NDEBUG) /* internal_malloc_site will be inlined */
# if defined(_WIN32) || defined(__CYGWIN__)
      void* stacktrace[] = { 0, 0, 0 };
# else
      void* stacktrace[] = { 0, 0 };
# endif
#else /* not inlined */
      void* stacktrace[] = { 0, 0, 0, 0 };
#endif
      const unsigned int size = sizeof(stacktrace) / sizeof(*stacktrace);
      if (size == libxsmm_backtrace(stacktrace, size)) {
        unsigned int i;
        *site = stacktrace[size-1];
        for (i = 0; i < *npools; ++i) {
          if (*site == internal_malloc_scratch_pool[i].site) {
            *hit = 1; return i;
          }
        }
        *hit = 0;
      }
      else {
        *site = 0;
        *hit = 0;
      }
    }
  }
  else
#endif
  {
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS)
    *npools = LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS;
#else
    *npools = 0;
#endif
    *site = 0;
    *hit = 1;
  }
  return 0;
}


LIBXSMM_API_DEFINITION void* libxsmm_scratch_malloc(size_t size, size_t alignment, const void* caller)
{
  void* result = 0;
  static int error_once = 0;
  LIBXSMM_INIT
#if !defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) || (0 >= (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  LIBXSMM_UNUSED(caller);
#else
  if (0 < libxsmm_scratch_pools) {
    unsigned int npools = 0, hit = 0, i;
    const unsigned int pool = internal_malloc_site(&npools, &hit, &caller);
    const size_t align_size = (0 == alignment ? libxsmm_alignment(size, alignment) : alignment);
    const size_t alloc_size = size + align_size - 1;
    size_t total_size = 0, local_size = 0, req_size = 0;

    for (i = pool; i < npools; ++i) {
      const size_t inuse_size = internal_malloc_scratch_pool[i].head - internal_malloc_scratch_pool[i].buffer;
      size_t info_size;
      /* memory information for scratch memory is documented to be unsupported; no extra/info size */
      if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(internal_malloc_scratch_pool[i].buffer, &info_size, 0/*flags*/, 0/*extra*/)) {
        total_size = info_size;
      }
      req_size = inuse_size + alloc_size;

      if (total_size < req_size) {
        if (0 == internal_malloc_scratch_pool[i].buffer) {
          LIBXSMM_LOCK_ACQUIRE(&libxsmm_lock_global);
          if (0 == internal_malloc_scratch_pool[i].buffer) {
            const double scratch_scale = 0 < libxsmm_scratch_scale ? libxsmm_scratch_scale : (LIBXSMM_MALLOC_SCRATCH_SCALE);
            const size_t minsize = (size_t)(scratch_scale * LIBXSMM_MAX(internal_malloc_scratch_pool[i].minsize, req_size));
            assert(0 == internal_malloc_scratch_pool[i].head/*sanity check*/);
            if (EXIT_SUCCESS == libxsmm_xmalloc((void**)&internal_malloc_scratch_pool[i].buffer, minsize, 0/*auto*/,
              LIBXSMM_MALLOC_FLAG_SCRATCH, 0/*extra*/, 0/*extra_size*/))
            {
              /* atomic update needed since modifications will also happen outside of this region */
              LIBXSMM_ATOMIC_STORE(&internal_malloc_scratch_pool[i].head,
                internal_malloc_scratch_pool[i].buffer, LIBXSMM_ATOMIC_SEQ_CST);
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
              internal_malloc_scratch_pool[i].site = caller;
#endif
              total_size = minsize;
            }
            else { /* fall-back to local allocation due to failed scratch memory allocation */
              if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
                fprintf(stderr, "LIBXSMM ERROR: failed to allocate scratch memory!\n");
              }
              local_size = size;
            }
            if (internal_malloc_scratch_pool[i].minsize < minsize) {
              LIBXSMM_ATOMIC_STORE(&internal_malloc_scratch_pool[i].minsize, minsize, LIBXSMM_ATOMIC_RELAXED);
            }
            if ((LIBXSMM_MALLOC_SCRATCH_INTERNAL) != caller) {
              LIBXSMM_ATOMIC_ADD_FETCH(&internal_malloc_scratch_nmallocs, 1, LIBXSMM_ATOMIC_RELAXED);
            }
          }
          else { /* fall-back to local memory allocation due to lock-contention */
            local_size = size;
          }
          LIBXSMM_LOCK_RELEASE(&libxsmm_lock_global);
          break;
        }
        else { /* check for next pool */
          if (0 != hit || 1 == (npools - i)) {
            local_size = size; /* fall-back to local memory allocation */
            break;
          }
        }
      }
      else if (0 == hit) { /* use foreign pool */
        break;
      }
      else { /* hit and fit */
        break;
      }
    }

    assert(0 != local_size || i < npools);
    if (0 == local_size) { /* draw from buffer */
      char *const next = (char*)LIBXSMM_ATOMIC_ADD_FETCH(
        (uintptr_t*)&internal_malloc_scratch_pool[i].head,
        alloc_size, LIBXSMM_ATOMIC_SEQ_CST);
      if (next <= (internal_malloc_scratch_pool[i].buffer + total_size)) {
        char *const aligned = LIBXSMM_ALIGN(next - alloc_size, align_size);
        LIBXSMM_ATOMIC_ADD_FETCH(
          &internal_malloc_scratch_pool[i].counter,
          1, LIBXSMM_ATOMIC_SEQ_CST);
        result = aligned;
      }
      else { /* scratch memory recently exhausted */
        local_size = size;
      }
    }

    if (0 != local_size) { /* fall-back to local memory allocation */
      if (i < npools && internal_malloc_scratch_pool[i].minsize < req_size) {
        LIBXSMM_ATOMIC_STORE(&internal_malloc_scratch_pool[i].minsize, req_size, LIBXSMM_ATOMIC_RELAXED);
      }
      if (EXIT_SUCCESS != libxsmm_xmalloc(&result, local_size, alignment,
        LIBXSMM_MALLOC_FLAG_SCRATCH, 0/*extra*/, 0/*extra_size*/) &&
        /* library code is expected to be mute */0 != libxsmm_verbosity &&
        1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: scratch memory fall-back failed!\n");
      }
      if ((LIBXSMM_MALLOC_SCRATCH_INTERNAL) != caller) {
        LIBXSMM_ATOMIC_ADD_FETCH(&internal_malloc_scratch_nmallocs, 1, LIBXSMM_ATOMIC_RELAXED);
      }
    }
  }
  else
#endif /*defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))*/
  if (EXIT_SUCCESS != libxsmm_xmalloc(&result, size, alignment,
    LIBXSMM_MALLOC_FLAG_SCRATCH, 0/*extra*/, 0/*extra_size*/) &&
    /* library code is expected to be mute */0 != libxsmm_verbosity &&
    1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: scratch memory allocation failed!\n");
  }

  return result;
}


LIBXSMM_API_DEFINITION void* libxsmm_malloc(size_t size)
{
  return libxsmm_aligned_malloc(size, 0/*auto*/);
}


LIBXSMM_API_INLINE int internal_scratch_free(const void* memory, unsigned int pool)
{
  int released = 0;
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  if (0 < libxsmm_scratch_pools) {
    const char *const scratch = internal_malloc_scratch_pool[pool].buffer;
    if (0 != scratch) { /* check if memory belongs to scratch domain or local domain */
      const char *const buffer = (const char*)memory;
      size_t total_size;

      if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(scratch, &total_size, 0/*flags*/, 0/*extra*/) &&
        scratch <= buffer && buffer < (scratch + total_size))
      {
        if (0 < LIBXSMM_ATOMIC_SUB_FETCH(&internal_malloc_scratch_pool[pool].counter, 1, LIBXSMM_ATOMIC_SEQ_CST)
          || internal_malloc_scratch_pool[pool].minsize <= total_size) /* reuse scratch domain */
        {
          /* TODO: document/check that allocation/deallocation adheres to linear/scoped allocator policy */
          LIBXSMM_ATOMIC_STORE(&internal_malloc_scratch_pool[pool].head,
            internal_malloc_scratch_pool[pool].buffer, LIBXSMM_ATOMIC_SEQ_CST);
        }
        else { /* reallocate scratch domain, TODO: ensure thread-safety */
          const char *const current = internal_malloc_scratch_pool[pool].buffer; /* current scratch */
          LIBXSMM_ATOMIC_STORE_ZERO(&internal_malloc_scratch_pool[pool].buffer, LIBXSMM_ATOMIC_SEQ_CST);
          LIBXSMM_ATOMIC_STORE_ZERO(&internal_malloc_scratch_pool[pool].head, LIBXSMM_ATOMIC_SEQ_CST);
          libxsmm_xfree(current);
        }
        released = 1;
      }
    }
  }
#else
  LIBXSMM_UNUSED(memory);
  LIBXSMM_UNUSED(pool);
#endif /*defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))*/
  return released;
}


LIBXSMM_API_DEFINITION void libxsmm_free(const void* memory)
{
  unsigned int npools, pool = 0, i = 0;
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  int hit = 0;
# if !defined(LIBXSMM_MALLOC_SCRATCH_XFREE)
  const void* site = 0;
  pool = internal_malloc_site(&npools, &hit, &site);
# endif
  if (0 != hit)
#endif
  {
    npools = internal_scratch_free(memory, pool);
  }
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  else { /* find scratch memory pool */
    npools = LIBXSMM_MIN(libxsmm_scratch_pools, LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS);
    for (; i < npools; ++i) {
      if (0 != internal_scratch_free(memory, i)) {
        i = npools + 1; /* break */
      }
    }
  }
#endif
  if (i == npools) { /* local */
    libxsmm_xfree(memory);
  }
}


LIBXSMM_API_DEFINITION void libxsmm_release_scratch(void)
{
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
  const unsigned int max_npools = LIBXSMM_MIN(libxsmm_scratch_pools, LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS);
  unsigned int i;
  for (i = 0; i < max_npools; ++i) { /* TODO: thread-safety */
    libxsmm_xfree(internal_malloc_scratch_pool[i].buffer);
    internal_malloc_scratch_pool[i].counter = 0;
    internal_malloc_scratch_pool[i].buffer = 0;
    internal_malloc_scratch_pool[i].head = 0;
  }
  if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    libxsmm_scratch_info scratch_info;
    if (EXIT_SUCCESS == libxsmm_get_scratch_info(&scratch_info) && 0 < scratch_info.npending) {
      fprintf(stderr, "LIBXSMM ERROR: %lu pending scratch-memory allocations!\n",
        (unsigned long int)scratch_info.npending);
    }
  }
#endif
}


LIBXSMM_API_DEFINITION int libxsmm_get_malloc_info(const void* memory, libxsmm_malloc_info* info)
{
  int result = EXIT_SUCCESS;
  if (0 != info) {
    size_t size;
    result = libxsmm_get_malloc_xinfo(memory, &size, 0/*flags*/, 0/*extra*/);
    if (EXIT_SUCCESS == result) {
      memset(info, 0, sizeof(libxsmm_malloc_info));
      info->size = size;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_get_scratch_info(libxsmm_scratch_info* info)
{
  int result = EXIT_SUCCESS;
  if (0 != info) {
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
    unsigned int i;
    memset(info, 0, sizeof(libxsmm_scratch_info));
    info->npending = internal_malloc_scratch_pool[0].counter;
    info->nmallocs = internal_malloc_scratch_nmallocs;
    info->npools = LIBXSMM_MIN(1, libxsmm_scratch_pools);

    if (
#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
      (LIBXSMM_MALLOC_SCRATCH_INTERNAL) != internal_malloc_scratch_pool[0].site &&
#endif
      (0 == internal_malloc_scratch_pool[0].buffer || EXIT_SUCCESS != libxsmm_get_malloc_xinfo(
            internal_malloc_scratch_pool[0].buffer, &info->size, 0/*flags*/, 0/*extra*/)))
    {
      info->size = internal_malloc_scratch_pool[0].minsize;
    }

#if defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))
    {
      const unsigned max_npools = LIBXSMM_MIN(libxsmm_scratch_pools, LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS);
      for (i = 1; i < max_npools; ++i) {
        if ((LIBXSMM_MALLOC_SCRATCH_INTERNAL) != internal_malloc_scratch_pool[i].site) {
          info->npools += (unsigned int)LIBXSMM_MIN(internal_malloc_scratch_pool[i].minsize, 1);
          info->npending += internal_malloc_scratch_pool[i].counter;
        }
      }
      if (0 != internal_malloc_scratch_pool[0].buffer) {
        for (i = 1; i < max_npools; ++i) {
          if ((LIBXSMM_MALLOC_SCRATCH_INTERNAL) != internal_malloc_scratch_pool[i].site) {
            size_t size;
            if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(
              internal_malloc_scratch_pool[i].buffer, &size, 0/*flags*/, 0/*extra*/))
            {
              info->size += size;
            }
          }
        }
      }
      else { /* approximate memory consumption by using minsize */
        for (i = 1; i < max_npools; ++i) {
          if ((LIBXSMM_MALLOC_SCRATCH_INTERNAL) != internal_malloc_scratch_pool[i].site) {
            info->size += internal_malloc_scratch_pool[i].minsize;
          }
        }
      }
    }
#endif /*defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (1 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))*/
#else
    memset(info, 0, sizeof(*info));
#endif /*defined(LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS) && (0 < (LIBXSMM_MALLOC_SCRATCH_MAX_NPOOLS))*/
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION unsigned int libxsmm_hash(const void* data, size_t size, unsigned int seed)
{
  LIBXSMM_INIT
  return libxsmm_crc32(data, size, seed);
}

