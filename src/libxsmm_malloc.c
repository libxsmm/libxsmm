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
/* Hans Pabst (Intel Corp.)
******************************************************************************/

/* must be defined *before* other files are included */
#if !defined(_GNU_SOURCE)
# define _GNU_SOURCE
#endif
#include "libxsmm_malloc.h"
#include "libxsmm_main.h"
#include <libxsmm_sync.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#if !defined(NDEBUG)
# include <string.h>
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
#endif /*defined(LIBXSMM_VTUNE)*/
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif
#if defined(LIBXSMM_PERF)
# include "libxsmm_perf.h"
#endif

#if !defined(LIBXSMM_MALLOC_ALIGNMAX)
# define LIBXSMM_MALLOC_ALIGNMAX (2 * 1024 *1024)
#endif
#if !defined(LIBXSMM_MALLOC_ALIGNFCT)
# define LIBXSMM_MALLOC_ALIGNFCT 8
#endif

/* perform low-level allocation even for small non-executable buffers */
#if !defined(LIBXSMM_MALLOC_MMAP)
/*# define LIBXSMM_MALLOC_MMAP*/
#endif


typedef struct LIBXSMM_RETARGETABLE internal_malloc_extra_type {
  /* also avoids warning about empty structure */
  void* reloc;
#if defined(LIBXSMM_VTUNE)
  unsigned int code_id;
#endif
} internal_malloc_extra_type;


typedef struct LIBXSMM_RETARGETABLE internal_malloc_info_type {
  void* pointer;
  size_t size;
  int flags;
  internal_malloc_extra_type internal;
} internal_malloc_info_type;


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


#if defined(LIBXSMM_VTUNE)
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_get_vtune_jitdesc(const volatile void* code,
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
# if (2 == LIBXSMM_VTUNE_JITVERSION)
  desc->module_name = "libxsmm.jit";
# endif
}
#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE internal_malloc_info_type* internal_malloc_info(const volatile void* memory)
{
  /* TODO: rely on correct memory information; implement checksum */
  return (internal_malloc_info_type*)(0 != memory ? (((const char*)memory) - sizeof(internal_malloc_info_type)) : 0);
}


LIBXSMM_API_DEFINITION int libxsmm_malloc_info(const volatile void* memory, size_t* size, int* flags, void** extra)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG)
  if (0 != size || 0 != extra)
#endif
  {
    if (0 != memory) {
      const internal_malloc_info_type *const info = internal_malloc_info(memory);
      assert(0 != info);
      if (size) *size = info->size;
      if (flags) *flags = info->flags;
      if (extra) *extra = info->pointer;
    }
    else {
      if (size) *size = 0;
      if (flags) *flags = 0;
      if (extra) *extra = 0;
    }
  }
#if !defined(NDEBUG)
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: attachment error for memory buffer %p!\n", memory);
    }
    result = EXIT_FAILURE;
  }
#endif
  assert(EXIT_SUCCESS == result);
  return result;
}


#if !defined(_WIN32)

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_mhint(void* buffer, size_t size)
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


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void* internal_xmap(const char* dir, size_t size, int flags, void** rx)
{
  void* result = MAP_FAILED;
  char filename[4096];
  int i = LIBXSMM_SNPRINTF(filename, sizeof(filename), "%s/XXXXXX.jit", dir);
  assert(0 != rx);
  if (0 <= i && i < (int)sizeof(filename)) {
    i = mkstemps(filename, 4);
    if (-1 != i && 0 == unlink(filename) && 0 == ftruncate(i, size)) {
      void *const xmap = mmap(0, size, PROT_READ | PROT_EXEC, flags | MAP_SHARED, i, 0);
      if (MAP_FAILED != xmap) {
        assert(0 != xmap);
        result = mmap(0, size, PROT_READ | PROT_WRITE, flags | MAP_SHARED, i, 0);
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


LIBXSMM_API_DEFINITION int libxsmm_xmalloc(void** memory, size_t size, int alignment,
  int flags, const void* extra, size_t extra_size)
{
  int result = EXIT_SUCCESS;
  if (memory) {
    if (0 < size) {
      const size_t internal_size = size + extra_size + sizeof(internal_malloc_info_type);
      size_t alloc_alignment = 0, alloc_size = 0;
      void *alloc_failed = 0, *buffer = 0, *reloc = 0;
#if !defined(NDEBUG)
      static int error_once = 0;
#endif
      flags |= LIBXSMM_MALLOC_FLAG_RW; /* normalize given flags since flags=0 is accepted as well */
#if !defined(LIBXSMM_MALLOC_MMAP)
      if (0 == (LIBXSMM_MALLOC_FLAG_X & flags) && 0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) {
        alloc_alignment = 0 <= alignment ? libxsmm_alignment(size, (size_t)alignment) : ((size_t)(-alignment));
        alloc_size = internal_size + alloc_alignment - 1;
        buffer = malloc(alloc_size);
      }
      else
#endif
      {
#if defined(_WIN32)
        const int xflags = (0 != (LIBXSMM_MALLOC_FLAG_X & flags) ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
        if ((LIBXSMM_MALLOC_ALIGNMAX * LIBXSMM_MALLOC_ALIGNFCT) > size) {
          alloc_alignment = 0 <= alignment ? libxsmm_alignment(size, (size_t)alignment) : ((size_t)(-alignment));
          alloc_size = internal_size + alloc_alignment - 1;
          buffer = VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, xflags);
        }
        else {
          HANDLE process_token;
          const SIZE_T alloc_alignmax = GetLargePageMinimum();
          /* respect user-requested alignment */
          alloc_alignment = 0 == alignment ? alloc_alignmax : libxsmm_lcm((size_t)LIBXSMM_ABS(alignment), alloc_alignmax);
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
            alloc_alignment = 0 <= alignment ? libxsmm_alignment(size, (size_t)alignment) : ((size_t)(-alignment));
            alloc_size = internal_size + alloc_alignment - 1;
            buffer = VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, xflags);
          }
        }
        if (alloc_failed != buffer) {
          flags |= LIBXSMM_MALLOC_FLAG_MMAP; /* select the corresponding deallocation */
        }
        else if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) { /* fall-back allocation */
          buffer = malloc(alloc_size);
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
        alloc_alignment = 0 <= alignment ? libxsmm_alignment(size, (size_t)alignment) : ((size_t)(-alignment));
        alloc_size = internal_size + alloc_alignment - 1;
        alloc_failed = MAP_FAILED;
        if (0 == (LIBXSMM_MALLOC_FLAG_X & flags)) {
# if defined(__APPLE__) && defined(__MACH__)
          buffer = mmap(0, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | xflags, -1, 0);
# else
          buffer = mmap(0, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | xflags, -1, 0);
# endif
        }
        else {
          static LIBXSMM_TLS int fallback = 0;
          if (0 == fallback) {
            buffer = internal_xmap("/tmp", alloc_size, xflags, &reloc);
            if (alloc_failed == buffer) fallback = 1;
          }
          if (1 == fallback) { /* 2nd try */
            buffer = internal_xmap(".", alloc_size, xflags, &reloc);
            if (alloc_failed == buffer) fallback = 2;
          }
          if (2 == fallback) { /* 3rd try */
            buffer = internal_xmap(getenv("HOME"), alloc_size, xflags, &reloc);
            if (alloc_failed == buffer) fallback = 3;
          }
          if (3 == fallback) { /* 4th try */
            buffer = internal_xmap(getenv("JITDUMPDIR"), alloc_size, xflags, &reloc);
            if (alloc_failed == buffer) fallback = 4;
          }
          if (4 == fallback) { /* never try again */
            buffer = alloc_failed;
            reloc = 0;
          }
        }
        if (alloc_failed != buffer) {
          assert(0 != buffer);
          flags |= LIBXSMM_MALLOC_FLAG_MMAP; /* select the corresponding deallocation */
        }
        else if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & flags)) { /* fall-back allocation */
          buffer = malloc(alloc_size);
          reloc = buffer;
        }
        if (MAP_FAILED != buffer && 0 != buffer) {
          internal_mhint(buffer, alloc_size);
        }
#endif
      }
      if (alloc_failed != buffer && /*fallback*/0 != buffer) {
        char *const aligned = LIBXSMM_ALIGN(((char*)buffer) + extra_size + sizeof(internal_malloc_info_type), alloc_alignment);
        internal_malloc_info_type *const info = (internal_malloc_info_type*)(aligned - sizeof(internal_malloc_info_type));
        assert((aligned + size) <= (((char*)buffer) + alloc_size));
#if !defined(NDEBUG) && !defined(_WIN32)
        memset(buffer, 0, alloc_size);
#endif
        if (0 < extra_size && 0 != extra) {
          const char *const src = (const char*)extra;
          char *const dst = (char*)buffer;
          size_t i;
#if defined(_MSC_VER) && (1900 <= _MSC_VER)
#         pragma warning(suppress: 6386)
#endif
          for (i = 0; i < extra_size; ++i) dst[i] = src[i];
        }
#if !defined(NDEBUG)
        else if (0 == extra && 0 != extra_size) {
          result = EXIT_FAILURE;
        }
#endif
        info->internal.reloc = reloc;
        info->pointer = buffer;
        info->size = size;
        info->flags = flags;
        *memory = aligned;
      }
      else {
#if !defined(NDEBUG) /* library code is expected to be mute */
        if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
          fprintf(stderr, "LIBXSMM: memory allocation error for size %llu with flags=%i!\n",
            (unsigned long long)alloc_size, flags);
        }
#endif
        result = EXIT_FAILURE;
      }
    }
    else {
      *memory = 0;
    }
  }
#if !defined(NDEBUG)
  else if (0 != size) {
    result = EXIT_FAILURE;
  }
#endif
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_xfree(const volatile void* memory)
{
  int result = EXIT_SUCCESS;
  if (memory) {
    /*const*/ internal_malloc_info_type *const info = internal_malloc_info(memory);
    assert((0 != info->pointer || 0 == info->size));
    if (0 == (LIBXSMM_MALLOC_FLAG_MMAP & info->flags)) {
      free(info->pointer);
    }
    else {
#if defined(LIBXSMM_VTUNE) || !defined(_WIN32)
      /*const*/ internal_malloc_extra_type *const internal = &info->internal;
      assert(0 != internal);
# if defined(LIBXSMM_VTUNE)
      if (0 != (LIBXSMM_MALLOC_FLAG_X & info->flags) && 0 != internal->code_id && iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
        iJIT_NotifyEvent(LIBXSMM_VTUNE_JIT_UNLOAD, &internal->code_id);
      }
# endif
#endif
#if defined(_WIN32)
      result = FALSE != VirtualFree(info->pointer, 0, MEM_RELEASE) ? EXIT_SUCCESS : EXIT_FAILURE;
#else /* defined(_WIN32) */
      {
        const size_t alloc_size = info->size + (((const char*)memory) - ((const char*)info->pointer));
        void *const buffer = info->pointer, *const reloc = internal->reloc;
        const int flags = info->flags;
        if (0 != munmap(buffer, alloc_size)) {
# if !defined(NDEBUG) /* library code is expected to be mute */
          static int error_once = 0;
          if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
            const char *const error_message = strerror(errno);
            fprintf(stderr, "LIBXSMM: %s (munmap error #%i for range %p+%llu)!\n",
              error_message, errno, buffer, (unsigned long long)alloc_size);
          }
# endif
          result = EXIT_FAILURE;
        }
        if (0 != (LIBXSMM_MALLOC_FLAG_X & flags) && EXIT_SUCCESS == result
         && 0 != reloc && MAP_FAILED != reloc
         && 0 != munmap(reloc, alloc_size))
        {
# if !defined(NDEBUG) /* library code is expected to be mute */
          static int error_once = 0;
          if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
            const char *const error_message = strerror(errno);
            fprintf(stderr, "LIBXSMM: %s (munmap error #%i for range %p+%llu)!\n",
              error_message, errno, reloc, (unsigned long long)alloc_size);
          }
# endif
          result = EXIT_FAILURE;
        }
      }
#endif
    }
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_malloc_attrib(void** memory, int flags, const char* name)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG)
  static int error_once = 0;
#endif
  if (0 != memory) {
    internal_malloc_info_type *const info = internal_malloc_info(*memory);
    internal_malloc_extra_type *const internal = &info->internal;
    void *const buffer = info->pointer;
    const size_t size = info->size;
    assert((0 != buffer || 0 == size) && 0 != internal);
    /* quietly keep the read permission, but eventually revoke write permissions */
    if (0 == (LIBXSMM_MALLOC_FLAG_W & flags) || 0 != (LIBXSMM_MALLOC_FLAG_X & flags)) {
      const int alignment = (int)(((const char*)(*memory)) - ((const char*)buffer));
      const size_t alloc_size = size + alignment;
      /* treat memory protection errors as soft error if the requested buffer is not executable */
      int soft_error = 0;
      if (0 == (LIBXSMM_MALLOC_FLAG_X & flags)) {
#if defined(_WIN32)
        /* TODO: implement memory protection under Microsoft Windows */
        LIBXSMM_UNUSED(internal); LIBXSMM_UNUSED(alloc_size); LIBXSMM_UNUSED(soft_error);
#else
        soft_error = mprotect(buffer, alloc_size/*entire memory region*/, PROT_READ);
#endif
      }
      else {
        void *const code_ptr =
#if !defined(_WIN32)
          0 != (LIBXSMM_MALLOC_FLAG_MMAP & flags) ? ((void*)(((char*)internal->reloc) + alignment)) :
#endif
          buffer;
        assert(0 != (LIBXSMM_MALLOC_FLAG_X & flags));
        if (name && *name) { /* profiler support requested */
          FILE *const code_file = fopen(name, "wb");
          if (0 != code_file) { /* dump byte-code into a file and print function pointer and filename */
            fprintf(stderr, "LIBXSMM-JIT-DUMP(ptr:file) %p : %s\n", code_ptr, name);
            fwrite(code_ptr, 1, size, code_file);
            fclose(code_file);
          }
#if defined(LIBXSMM_VTUNE)
          if (iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
            LIBXSMM_VTUNE_JIT_DESC_TYPE vtune_jit_desc;
            const unsigned int code_id = iJIT_GetNewMethodID();
            internal_get_vtune_jitdesc(code_ptr, code_id, size, name, &vtune_jit_desc);
            iJIT_NotifyEvent(LIBXSMM_VTUNE_JIT_LOAD, &vtune_jit_desc);
            internal->code_id = code_id;
          }
          else {
            internal->code_id = 0;
          }
#endif
#if defined(LIBXSMM_PERF)
          /* If jitting is enabled and a valid name is given, emit information for perf.
           * In jitdump case this needs to be done after mprotect as it gets overwritten
           * otherwise. */
          libxsmm_perf_dump_code(code_ptr, size, name);
#endif
        }
        if (0 != (LIBXSMM_MALLOC_FLAG_MMAP & flags)) {
#if defined(_WIN32)
          /* TODO: implement memory protection under Microsoft Windows */
#else
          *memory = code_ptr; /* relocate */
          info->pointer = internal->reloc;
          internal->reloc = 0;
          if (0 != buffer && MAP_FAILED != buffer) {
            soft_error = munmap(buffer, alloc_size);
          }
#endif
        }
#if !defined(_WIN32)
        else { /* malloc-based fall-back */
          /* mprotect memory region according to the requested flags */
          soft_error = mprotect(buffer, alloc_size/*entire memory region*/, PROT_READ | PROT_EXEC);
          if (0/*ok*/ != soft_error) result = EXIT_FAILURE; /* error cannot be ignored */
        }
#endif
      }
#if !defined(NDEBUG) /* library code is expected to be mute */
      if (0/*ok*/ != soft_error && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
        const char *const error_message = strerror(errno);
        fprintf(stderr, "LIBXSMM: %s (error #%i for range %p+%llu with flags=%i)!\n",
          error_message, errno, buffer, (unsigned long long)alloc_size, flags);
      }
#endif
    }
  }
  else {
#if !defined(NDEBUG) /* library code is expected to be mute */
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: libxsmm_malloc_attrib failed because NULL cannot be attributed!\n");
    }
#endif
    result = EXIT_FAILURE;
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION void* libxsmm_aligned_malloc(size_t size, int alignment)
{
  void* result = 0;
  return 0 == libxsmm_xmalloc(&result, size, alignment, LIBXSMM_MALLOC_FLAG_DEFAULT,
    0/*extra*/, 0/*extra_size*/) ? result : 0;
}


LIBXSMM_API_DEFINITION void* libxsmm_malloc(size_t size)
{
  return libxsmm_aligned_malloc(size, 0/*auto*/);
}


LIBXSMM_API_DEFINITION void libxsmm_free(const volatile void* memory)
{
  libxsmm_xfree(memory);
}

