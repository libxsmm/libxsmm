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
# define LIBXSMM_ALLOC_ALIGNFCT 2
#endif
#if !defined(LIBXSMM_ALLOC_MMAP)
/*# define LIBXSMM_ALLOC_MMAP*/
#endif


typedef struct LIBXSMM_RETARGETABLE internal_alloc_extra_type {
#if defined(LIBXSMM_VTUNE)
  unsigned int code_id;
#else /* avoid warning about empty struct */
  int dummy;
#endif
} internal_alloc_extra_type;


typedef struct LIBXSMM_RETARGETABLE internal_alloc_info_type {
  void* pointer;
  unsigned int size;
  int flags;
  internal_alloc_extra_type internal;
} internal_alloc_info_type;


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_gcd(unsigned int a, unsigned int b)
{
  while (0 != b) {
    const unsigned int r = a % b;
    a = b;
    b = r;
  }
  return a;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_lcm(unsigned int a, unsigned int b)
{
  return (a * b) / libxsmm_gcd(a, b);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_alignment(unsigned int size, unsigned int alignment)
{
  unsigned int result = sizeof(void*);
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
  unsigned int code_id, unsigned int code_size, const char* code_name,
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


LIBXSMM_INLINE LIBXSMM_RETARGETABLE int internal_alloc_info(const void* memory, unsigned int* size, int* flags,
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


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_alloc_info(const void* memory, unsigned int* size, int* flags, void** extra)
{
  return internal_alloc_info(memory, size, flags, extra, 0/*internal*/);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_allocate(void** memory, unsigned int size, unsigned int alignment,
  int flags, const void* extra, unsigned int extra_size)
{
  int result = EXIT_SUCCESS;
  if (memory) {
    if (0 < size) {
      const unsigned int auto_alignment = libxsmm_alignment(size, alignment);
      const unsigned int alloc_size = size + extra_size + sizeof(internal_alloc_info_type) + auto_alignment - 1;
      void* alloc_failed = 0;
      char* buffer = 0;
#if !defined(NDEBUG)
      static LIBXSMM_TLS int alloc_error = 0;
#endif
#if !defined(LIBXSMM_ALLOC_MMAP)
      if (0 == flags || LIBXSMM_ALLOC_FLAG_DEFAULT == flags) {
        buffer = malloc(alloc_size);
      }
      else
#endif
      {
#if defined(_WIN32)
        const int xflags = (0 != (LIBXSMM_ALLOC_FLAG_X & flags) ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE);
        buffer = (char*)VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, xflags);
#else
        const int xflags = (0 != (LIBXSMM_ALLOC_FLAG_X & flags) ? (PROT_READ | PROT_WRITE | PROT_EXEC) : (PROT_READ | PROT_WRITE));
        alloc_failed = MAP_FAILED;
        buffer = (char*)mmap(0, alloc_size, xflags,
# if defined(__APPLE__) && defined(__MACH__)
          MAP_ANON | MAP_PRIVATE,
# elif !defined(__CYGWIN__)
          MAP_ANONYMOUS | MAP_PRIVATE | MAP_32BIT,
# else
          MAP_ANONYMOUS | MAP_PRIVATE,
# endif
          -1, 0);
# if defined(MADV_NOHUGEPAGE)
        /* disable THP for smaller allocations; req. Linux kernel 2.6.38 (or higher) */
        if (LIBXSMM_ALLOC_ALIGNMAX > alloc_size && alloc_failed != buffer) {
#   if defined(NDEBUG)
          /* proceed even in case of an error, we then just take what we got (THP) */
          madvise(buffer, alloc_size, MADV_NOHUGEPAGE);
#   else /* library code is expected to be mute */
          if (0 != madvise(buffer, alloc_size, MADV_NOHUGEPAGE)) {
            static LIBXSMM_TLS int madvise_error = 0;
            if (0 == madvise_error) {
              fprintf(stderr, "LIBXSMM: %s (madvise error #%i for range %p+%u)!\n",
                strerror(errno), errno, buffer, alloc_size);
              madvise_error = 1;
            }
          }
#   endif /*defined(NDEBUG)*/
        }
#   if !defined(NDEBUG) /* library code is expected to be mute */
        else if (alloc_failed == buffer && 0 == alloc_error) {
          fprintf(stderr, "LIBXSMM: %s (mmap error #%i for size %u with flags=%i)!\n",
            strerror(errno), errno, alloc_size, xflags);
          alloc_error = 1;
        }
#   endif
# elif !(defined(__APPLE__) && defined(__MACH__)) && !defined(__CYGWIN__)
        LIBXSMM_MESSAGE("================================================================================")
        LIBXSMM_MESSAGE("LIBXSMM: Adjusting THP is unavailable due to C89 or kernel older than 2.6.38!")
        LIBXSMM_MESSAGE("================================================================================")
# endif /*MADV_NOHUGEPAGE*/
#endif
      }
      if (alloc_failed != buffer) {
        char *const aligned = LIBXSMM_ALIGN(buffer + extra_size + sizeof(internal_alloc_info_type), auto_alignment);
        internal_alloc_info_type *const info = (internal_alloc_info_type*)(aligned - sizeof(internal_alloc_info_type));
        assert((aligned + size) <= (buffer + alloc_size));
#if !defined(NDEBUG)
        memset(buffer, 0, alloc_size);
#endif
        if (0 < extra_size && 0 != extra) {
          const char *const src = (const char*)extra;
          unsigned int i;
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
          fprintf(stderr, "LIBXSMM: memory allocation error for size %u with flags=%i!\n", alloc_size, flags);
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


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_deallocate(const void* memory)
{
  int result = EXIT_SUCCESS;
  if (memory) {
    internal_alloc_extra_type* internal = 0;
    unsigned int size = 0;
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
#if defined(_WIN32)
      result = FALSE != VirtualFree(buffer, 0, MEM_RELEASE) ? EXIT_SUCCESS : EXIT_FAILURE;
#else
      const unsigned int alloc_size = size + (((const char*)memory) - ((const char*)buffer));
      if (0 != munmap(buffer, alloc_size)) {
# if !defined(NDEBUG) /* library code is expected to be mute */
        static LIBXSMM_TLS int munmap_error = 0;
        if (0 == munmap_error) {
          fprintf(stderr, "LIBXSMM: %s (munmap error #%i for range %p+%u)!\n",
            strerror(errno), errno, buffer, alloc_size);
          munmap_error = 1;
        }
# endif
        result = EXIT_FAILURE;
      }
#endif
#if defined(LIBXSMM_VTUNE)
      assert(0 != internal);
      if (0 != (LIBXSMM_ALLOC_FLAG_X & flags) && 0 != internal->code_id && iJIT_SAMPLING_ON == iJIT_IsProfilingActive()) {
        iJIT_NotifyEvent(LIBXSMM_VTUNE_JIT_UNLOAD, &internal->code_id);
      }
#endif
    }
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_alloc_attribute(const void* memory, int flags, const char* name)
{
  void* buffer = 0;
  unsigned int size = 0;
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
#if defined(_WIN32) /*TODO: implementation for Microsoft Windows*/
    LIBXSMM_UNUSED(memory); LIBXSMM_UNUSED(flags); LIBXSMM_UNUSED(name);
#else
    const unsigned int alloc_size = ((const char*)memory) - ((const char*)buffer);
    int xflags = PROT_READ | PROT_WRITE | PROT_EXEC;
    if (0 != (LIBXSMM_ALLOC_FLAG_W & flags)) xflags &= ~PROT_WRITE;
    if (0 != (LIBXSMM_ALLOC_FLAG_X & flags)) xflags &= ~PROT_EXEC;
    if (0/*ok*/ != mprotect(buffer, alloc_size/*entire memory region*/, xflags)) {
# if !defined(NDEBUG) /* library code is expected to be mute */
      if (0 == revoke_error) {
        fprintf(stderr, "LIBXSMM: %s (mprotect error #%i for range %p+%u with flags=%i)!\n",
          strerror(errno), errno, buffer, alloc_size, xflags);
        revoke_error = 1;
      }
# endif
      result = EXIT_FAILURE;
    }
#endif
#if (!defined(NDEBUG) && defined(_DEBUG)) || defined(LIBXSMM_VTUNE)
    if (0 != (LIBXSMM_ALLOC_FLAG_X & alloc_flags) && EXIT_SUCCESS == result && name && *name) {
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
#endif
  }
  assert(EXIT_SUCCESS == result);
  return result;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void* libxsmm_malloc(unsigned int size)
{
  void* result = 0;
  return 0 == libxsmm_allocate(&result, size, 0/*auto*/, LIBXSMM_ALLOC_FLAG_DEFAULT,
    0/*extra*/, 0/*extra_size*/) ? result : 0;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_free(const void* memory)
{
  libxsmm_deallocate(memory);
}

