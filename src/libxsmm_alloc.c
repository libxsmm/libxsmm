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
#include "libxsmm_alloc.h"
#include "libxsmm_sync.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#if !defined(NDEBUG)
# include <string.h>
# include <stdio.h>
#endif
#if defined(_WIN32)
# include <windows.h>
#else
# include <sys/mman.h>
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

#if !defined(LIBXSMM_ALLOC_ALIGNMAX)
# define LIBXSMM_ALLOC_ALIGNMAX (2 * 1024 *1024)
#endif
#if !defined(LIBXSMM_ALLOC_ALIGNFCT)
# define LIBXSMM_ALLOC_ALIGNFCT 8
#endif
#if !defined(LIBXSMM_ALLOC_MMAP)
/*# define LIBXSMM_ALLOC_MMAP*/
#endif


typedef struct LIBXSMM_RETARGETABLE internal_alloc_extra_type {
#if defined(LIBXSMM_VTUNE)
  unsigned int code_id;
#else /* avoid warning about empty structure */
  int dummy;
#endif
} internal_alloc_extra_type;


typedef struct LIBXSMM_RETARGETABLE internal_alloc_info_type {
  void* pointer;
  size_t size;
  int flags;
  internal_alloc_extra_type internal;
} internal_alloc_info_type;


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
  if ((LIBXSMM_ALLOC_ALIGNFCT * LIBXSMM_ALLOC_ALIGNMAX) <= size) {
    result = libxsmm_lcm(0 == alignment ? (LIBXSMM_ALIGNMENT) : libxsmm_lcm(alignment, LIBXSMM_ALIGNMENT), LIBXSMM_ALLOC_ALIGNMAX);
  }
  else {
    if ((LIBXSMM_ALLOC_ALIGNFCT * LIBXSMM_ALIGNMENT) <= size) {
      result = (0 == alignment ? (LIBXSMM_ALIGNMENT) : libxsmm_lcm(alignment, LIBXSMM_ALIGNMENT));
    }
    else if (0 != alignment) {
      result = libxsmm_lcm(alignment, result);
    }
  }
  return result;
}


#if defined(LIBXSMM_VTUNE)
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_get_vtune_jitdesc(const void* code,
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


LIBXSMM_INLINE LIBXSMM_RETARGETABLE int internal_alloc_info(const void* memory, size_t* size, int* flags,
  void** extra, internal_alloc_extra_type** internal)
{
  int result = EXIT_SUCCESS;
#if !defined(NDEBUG)
  if (0 != size || 0 != extra)
#endif
  {
    if (0 != memory) {
      internal_alloc_info_type *const info = (internal_alloc_info_type*)(((const char*)memory) - sizeof(internal_alloc_info_type));
      if (size) *size = info->size;
      if (flags) *flags = info->flags;
      if (extra) *extra = info->pointer;
      if (internal) *internal = &info->internal;
    }
    else {
      if (size) *size = 0;
      if (flags) *flags = 0;
      if (extra) *extra = 0;
      if (internal) *internal = 0;
    }
  }
#if !defined(NDEBUG)
  else {
    static LIBXSMM_TLS int info_error = 0;
    if (0 == info_error) {
      fprintf(stderr, "LIBXSMM: attachment error for memory buffer %p!\n", memory);
      info_error = 1;
    }
    result = EXIT_FAILURE;
  }
#endif
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_alloc_info(const void* memory, size_t* size, int* flags, void** extra)
{
  return internal_alloc_info(memory, size, flags, extra, 0/*internal*/);
}


LIBXSMM_API_DEFINITION int libxsmm_allocate(void** memory, size_t size, size_t alignment,
  int flags, const void* extra, size_t extra_size)
{
  int result = EXIT_SUCCESS;
  if (memory) {
    if (0 < size) {
      const size_t internal_size = size + extra_size + sizeof(internal_alloc_info_type);
      size_t alloc_alignment = 0, alloc_size = 0;
      void* alloc_failed = 0;
      char* buffer = 0;
#if !defined(NDEBUG)
      static LIBXSMM_TLS int alloc_error = 0;
#endif
#if !defined(LIBXSMM_ALLOC_MMAP)
      if (0 == flags || LIBXSMM_ALLOC_FLAG_DEFAULT == flags) {
        alloc_alignment = libxsmm_alignment(size, alignment);
        alloc_size = internal_size + alloc_alignment - 1;
        buffer = (char*)malloc(alloc_size);
      }
      else
#endif
      {
#if defined(_WIN32)
        const int xflags = (0 != (LIBXSMM_ALLOC_FLAG_X & flags) ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
        if ((LIBXSMM_ALLOC_ALIGNMAX * LIBXSMM_ALLOC_ALIGNFCT) > size) {
          alloc_alignment = libxsmm_alignment(size, alignment);
          alloc_size = internal_size + alloc_alignment - 1;
          buffer = (char*)VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, xflags);
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
                buffer = (char*)VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, xflags);
              }
              tp.Privileges[0].Attributes = 0; /* disable privilege */
              AdjustTokenPrivileges(process_token, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
            }
            CloseHandle(process_token);
          }

          if (alloc_failed == buffer) { /* retry allocation with regular page size */
            alloc_alignment = libxsmm_alignment(size, alignment);
            alloc_size = internal_size + alloc_alignment - 1;
            buffer = (char*)VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, xflags);
          }
        }
# if !defined(NDEBUG) /* library code is expected to be mute */
        if (alloc_failed == buffer && 0 == alloc_error) { /* OS-specific error message */
          fprintf(stderr, "LIBXSMM: VirtualAlloc error #%lu for size %llu with flags=%i!\n",
            (unsigned long)GetLastError(), (unsigned long long)alloc_size, xflags);
          alloc_error = 1;
        }
# endif
#else
        const int xflags = (0 != (LIBXSMM_ALLOC_FLAG_X & flags) ? (PROT_READ | PROT_WRITE | PROT_EXEC) : (PROT_READ | PROT_WRITE));
        alloc_alignment = libxsmm_alignment(size, alignment);
        alloc_size = internal_size + alloc_alignment - 1;
        alloc_failed = MAP_FAILED;
        buffer = (char*)mmap(0, alloc_size, xflags,
# if defined(__APPLE__) && defined(__MACH__)
          MAP_PRIVATE | MAP_ANON
# else
          MAP_PRIVATE | MAP_ANONYMOUS
# endif
# if defined(MAP_NORESERVE)
          | MAP_NORESERVE
# endif
# if defined(MAP_HUGETLB)
          | ((LIBXSMM_ALLOC_ALIGNMAX * LIBXSMM_ALLOC_ALIGNFCT) > size ? 0 : MAP_HUGETLB)
# endif
# if defined(MAP_32BIT)
          | MAP_32BIT
# endif
          , -1, 0);
        if (alloc_failed != buffer) {
# if !defined(NDEBUG)
          if (0 !=
# endif
          /* proceed after failed madvise (even in case of an error; take what we got from mmap) */
          madvise(buffer, alloc_size, MADV_RANDOM
#   if defined(MADV_NOHUGEPAGE) /* if not available, we then take what we got (THP) */
            | ((LIBXSMM_ALLOC_ALIGNMAX * LIBXSMM_ALLOC_ALIGNFCT) > size ? MADV_NOHUGEPAGE : 0)
#   endif
#   if defined(MADV_DONTDUMP)
            | ((LIBXSMM_ALLOC_ALIGNMAX * LIBXSMM_ALLOC_ALIGNFCT) > size ? 0 : MADV_DONTDUMP)
#   endif
          )
# if !defined(NDEBUG)
          /* library code is expected to be mute */) {
            static LIBXSMM_TLS int madvise_error = 0;
            if (0 == madvise_error) {
              fprintf(stderr, "LIBXSMM: %s (madvise error #%i for range %p+%llu)!\n",
                strerror(errno), errno, buffer, (unsigned long long)alloc_size);
              madvise_error = 1;
            }
          }
# else
          ;
# endif
        }
# if !defined(NDEBUG) /* library code is expected to be mute */
        else if (alloc_failed == buffer && 0 == alloc_error) {
          fprintf(stderr, "LIBXSMM: %s (mmap error #%i for size %llu with flags=%i)!\n",
            strerror(errno), errno, (unsigned long long)alloc_size, xflags);
          alloc_error = 1;
        }
# endif
# if !defined(MADV_NOHUGEPAGE) && !(defined(__APPLE__) && defined(__MACH__)) && !defined(__CYGWIN__)
        LIBXSMM_MESSAGE("================================================================================")
        LIBXSMM_MESSAGE("LIBXSMM: Adjusting THP is unavailable due to C89 or kernel older than 2.6.38!")
        LIBXSMM_MESSAGE("================================================================================")
# endif /*MADV_NOHUGEPAGE*/
#endif
      }
      if (alloc_failed != buffer) {
        char *const aligned = LIBXSMM_ALIGN(buffer + extra_size + sizeof(internal_alloc_info_type), alloc_alignment);
        internal_alloc_info_type *const info = (internal_alloc_info_type*)(aligned - sizeof(internal_alloc_info_type));
        assert((aligned + size) <= (buffer + alloc_size));
#if !defined(NDEBUG) && !defined(_WIN32)
        memset(buffer, 0, alloc_size);
#endif
        if (0 < extra_size && 0 != extra) {
          const char *const src = (const char*)extra;
          size_t i;
#if (1900 <= _MSC_VER)
#         pragma warning(suppress: 6386)
#endif
          for (i = 0; i < extra_size; ++i) buffer[i] = src[i];
        }
#if !defined(NDEBUG)
        else if (0 == extra && 0 != extra_size) {
          result = EXIT_FAILURE;
        }
#endif
        info->pointer = buffer;
        info->size = size;
        info->flags = (0 == flags ? LIBXSMM_ALLOC_FLAG_DEFAULT : (0 != (LIBXSMM_ALLOC_FLAG_X & flags)
          /* normalize given flags */
          ? LIBXSMM_ALLOC_FLAG_RWX
          : LIBXSMM_ALLOC_FLAG_RW));
        *memory = aligned;
      }
      else {
#if !defined(NDEBUG) /* library code is expected to be mute */
        if (0 == alloc_error) {
          fprintf(stderr, "LIBXSMM: memory allocation error for size %llu with flags=%i!\n",
            (unsigned long long)alloc_size, flags);
          alloc_error = 1;
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


LIBXSMM_API_DEFINITION int libxsmm_deallocate(const void* memory)
{
  int result = EXIT_SUCCESS;
  if (memory) {
    internal_alloc_extra_type* internal = 0;
    size_t size = 0;
    void* buffer = 0;
    int flags = 0;
    result = internal_alloc_info(memory, &size, &flags, &buffer, &internal);
#if !defined(LIBXSMM_ALLOC_MMAP)
    if (LIBXSMM_ALLOC_FLAG_DEFAULT == flags && EXIT_SUCCESS == result) {
      free(buffer);
    }
    else
#endif
    if (EXIT_SUCCESS == result) {
#if defined(LIBXSMM_VTUNE)
      assert(0 != internal);
      if (0 != (LIBXSMM_ALLOC_FLAG_X & flags) && 0 != internal->code_id && iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
        iJIT_NotifyEvent(LIBXSMM_VTUNE_JIT_UNLOAD, &internal->code_id);
      }
#endif
#if defined(_WIN32)
      result = FALSE != VirtualFree(buffer, 0, MEM_RELEASE) ? EXIT_SUCCESS : EXIT_FAILURE;
#else
      const size_t alloc_size = size + (((const char*)memory) - ((const char*)buffer));
      if (0 != munmap(buffer, alloc_size)) {
# if !defined(NDEBUG) /* library code is expected to be mute */
        static LIBXSMM_TLS int munmap_error = 0;
        if (0 == munmap_error) {
          fprintf(stderr, "LIBXSMM: %s (munmap error #%i for range %p+%llu)!\n",
            strerror(errno), errno, buffer, (unsigned long long)alloc_size);
          munmap_error = 1;
        }
# endif
        result = EXIT_FAILURE;
      }
#endif
    }
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_alloc_attribute(const void* memory, int flags, const char* name)
{
  void* buffer = 0;
  size_t size = 0;
#if (!defined(NDEBUG) && defined(_DEBUG)) || defined(LIBXSMM_VTUNE)
  int alloc_flags = 0;
  internal_alloc_extra_type* internal = 0;
  int result = internal_alloc_info(memory, &size, &alloc_flags, &buffer, &internal);
#else
  int result = internal_alloc_info(memory, &size, 0/*flags*/, &buffer, 0/*internal*/);
#endif
#if !defined(NDEBUG)
  static LIBXSMM_TLS int revoke_error = 0;
#endif
  if (0 != buffer && EXIT_SUCCESS == result) {
#if (!defined(NDEBUG) && defined(_DEBUG)) || defined(LIBXSMM_VTUNE)
    if (0 != (LIBXSMM_ALLOC_FLAG_X & alloc_flags) && name && *name) {
# if !defined(NDEBUG) && defined(_DEBUG) /* dump byte-code into a file */
      FILE *const code_file = fopen(name, "wb");
      if (0 != code_file) {
        fwrite(memory, 1, size, code_file);
        fclose(code_file);
      }
# endif
# if defined(LIBXSMM_VTUNE)
      assert(0 != internal);
      if (iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
        LIBXSMM_VTUNE_JIT_DESC_TYPE vtune_jit_desc;
        const unsigned int code_id = iJIT_GetNewMethodID();
        internal_get_vtune_jitdesc(memory, code_id, size, name, &vtune_jit_desc);
        iJIT_NotifyEvent(LIBXSMM_VTUNE_JIT_LOAD, &vtune_jit_desc);
        internal->code_id = code_id;
      }
      else {
        internal->code_id = 0;
      }
# endif
    }
#else /* no VTune JIT Profiling and no debug code */
    LIBXSMM_UNUSED(name);
#endif
    { /* protect memory region according to the requested flags */
#if defined(_WIN32) /*TODO: implementation for Microsoft Windows*/
      LIBXSMM_UNUSED(memory); LIBXSMM_UNUSED(flags); LIBXSMM_UNUSED(name);
#else
      const size_t alloc_size = size + (((const char*)memory) - ((const char*)buffer));
      int xflags = PROT_READ | PROT_WRITE | PROT_EXEC;
      if (0 != (LIBXSMM_ALLOC_FLAG_W & flags)) xflags &= ~PROT_WRITE;
      if (0 != (LIBXSMM_ALLOC_FLAG_X & flags)) xflags &= ~PROT_EXEC;
      if (0/*ok*/ != mprotect(buffer, alloc_size/*entire memory region*/, xflags)) {
# if !defined(NDEBUG) /* library code is expected to be mute */
        if (0 == revoke_error) {
          fprintf(stderr, "LIBXSMM: %s (mprotect error #%i for range %p+%llu with flags=%i)!\n",
            strerror(errno), errno, buffer, (unsigned long long)alloc_size, xflags);
          revoke_error = 1;
        }
# endif
        result = EXIT_FAILURE;
      }
#endif
    }
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


#if defined(LIBXSMM_BUILD)

LIBXSMM_API_DEFINITION void* libxsmm_malloc(size_t size)
{
  void* result = 0;
  return 0 == libxsmm_allocate(&result, size, 0/*auto*/, LIBXSMM_ALLOC_FLAG_DEFAULT,
    0/*extra*/, 0/*extra_size*/) ? result : 0;
}


LIBXSMM_API_DEFINITION void libxsmm_free(const void* memory)
{
  libxsmm_deallocate(memory);
}

#endif /*defined(LIBXSMM_BUILD)*/

