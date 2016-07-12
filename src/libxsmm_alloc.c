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


typedef struct LIBXSMM_RETARGETABLE internal_alloc_info_type {
  void* pointer;
  unsigned int size;
#if !defined(LIBXSMM_ALLOC_MMAP)
  int flags;
#endif
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


LIBXSMM_INLINE LIBXSMM_RETARGETABLE int internal_alloc_info(const void* memory, unsigned int* size, void** extra, int* flags)
{
#if !defined(NDEBUG)
  if (0 != size || 0 != extra)
#endif
  {
    if (0 != memory) {
      const internal_alloc_info_type *const info = (const internal_alloc_info_type*)(((const char*)memory) - sizeof(internal_alloc_info_type));
      if (size) *size = info->size;
      if (extra) *extra = info->pointer;
#if defined(LIBXSMM_ALLOC_MMAP)
      if (flags) *flags = 0;
#else
      if (flags) *flags = info->flags;
#endif
    }
    else {
      if (size) *size = 0;
      if (extra) *extra = 0;
      if (flags) *flags = 0;
    }
  }
#if !defined(NDEBUG)
  else {
    return EXIT_FAILURE;
  }
#endif
  return EXIT_SUCCESS;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_alloc_info(const void* memory, unsigned int* size, void** extra)
{
  return internal_alloc_info(memory, size, extra, 0/*flags*/);
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
#if !defined(LIBXSMM_ALLOC_MMAP)
      if (0 == flags || LIBXSMM_ALLOC_FLAG_DEFAULT == flags) {
        buffer = malloc(alloc_size);
      }
      else
#endif
      {
#if defined(_WIN32)
        switch (flags) {
          case LIBXSMM_ALLOC_FLAG_RW:
          case 0: {
            buffer = (char*)VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
          } break;
          case LIBXSMM_ALLOC_FLAG_RWX: {
            buffer = (char*)VirtualAlloc(0, alloc_size, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
          } break;
        }
#else
        const int xflags = (LIBXSMM_ALLOC_FLAG_RW == flags || 0 == flags) ? (PROT_READ | PROT_WRITE)
          : (LIBXSMM_ALLOC_FLAG_RWX == flags ? (PROT_READ | PROT_WRITE | PROT_EXEC) : PROT_NONE);
        alloc_failed = MAP_FAILED;
        if (PROT_NONE != xflags) {
          buffer = (char*)mmap(0, alloc_size, xflags,
# if defined(__APPLE__) && defined(__MACH__)
            MAP_ANON,
# elif !defined(__CYGWIN__)
            MAP_ANONYMOUS | MAP_PRIVATE | MAP_32BIT,
# else
            MAP_ANONYMOUS | MAP_PRIVATE,
# endif
            -1, 0);
# if defined(MADV_NOHUGEPAGE)
          /* disable THP for smaller allocations; req. Linux kernel 2.6.38 (or higher) */
          if (LIBXSMM_ALLOC_ALIGNMAX > alloc_size) {
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
# elif !(defined(__APPLE__) && defined(__MACH__)) && !defined(__CYGWIN__)
          LIBXSMM_MESSAGE("================================================================================")
          LIBXSMM_MESSAGE("LIBXSMM: Adjusting THP is unavailable due to C89 or kernel older than 2.6.38!")
          LIBXSMM_MESSAGE("================================================================================")
# endif /*MADV_NOHUGEPAGE*/
        }
        else {
          buffer = alloc_failed;
        }
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
#if !defined(LIBXSMM_ALLOC_MMAP)
        info->flags = 0 == flags ? LIBXSMM_ALLOC_FLAG_DEFAULT : flags;
#endif
        *memory = aligned;
      }
      else {
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
    unsigned int size = 0;
    void* buffer = 0;
#if !defined(LIBXSMM_ALLOC_MMAP)
    int flags = 0;
    internal_alloc_info(memory, &size, &buffer, &flags);
    if (LIBXSMM_ALLOC_FLAG_DEFAULT == flags) {
      free(buffer);
    }
    else
#else
    internal_alloc_info(memory, &size, &buffer, 0/*flags*/);
#endif
    {
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
    }
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

