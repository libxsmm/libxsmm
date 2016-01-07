/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#include "libxsmm_trace.h"

#if defined(LIBXSMM_OFFLOAD_BUILD)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#if !defined(NDEBUG)
#include <errno.h>
#endif
#if defined(_WIN32) || defined(__CYGWIN__)
# include <Windows.h>
# include <DbgHelp.h>
#else
# include <execinfo.h>
# include <sys/stat.h>
# include <sys/mman.h>
# include <pthread.h>
# include <unistd.h>
# include <fcntl.h>
#endif
#if defined(LIBXSMM_OFFLOAD_BUILD)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_TRACE_MAXDEPTH) || 0 >= (LIBXSMM_TRACE_MAXDEPTH)
# undef LIBXSMM_TRACE_MAXDEPTH
# define LIBXSMM_TRACE_MAXDEPTH 16
#endif
#if !defined(LIBXSMM_TRACE_SYMBOLSIZE) || 0 >= (LIBXSMM_TRACE_SYMBOLSIZE)
# undef LIBXSMM_TRACE_SYMBOLSIZE
# define LIBXSMM_TRACE_SYMBOLSIZE 256
#endif
#if !defined(LIBXSMM_TRACE_STDATOMIC) && defined(__GNUC__) && \
  (40704 <= (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__))
# define LIBXSMM_TRACE_STDATOMIC
#endif


#if defined(_WIN32) || defined(__CYGWIN__)
# define LIBXSMM_TRACE_MINDEPTH 5
LIBXSMM_RETARGETABLE LIBXSMM_ALIGNED(volatile LONG libxsmm_trace_initialized, 32) = -2;
#else
# define LIBXSMM_TRACE_MINDEPTH 4
LIBXSMM_RETARGETABLE LIBXSMM_ALIGNED(int libxsmm_trace_initialized, 32) = -2;
LIBXSMM_RETARGETABLE pthread_key_t libxsmm_trace_key = 0;

#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_delete(void* value)
{
  int fd;
  assert(value);
  fd = *((int*)value);

#if defined(NDEBUG)
  munmap(value, LIBXSMM_TRACE_SYMBOLSIZE);
#else /* library code is expected to be mute */
  if (0 != munmap(value, LIBXSMM_TRACE_SYMBOLSIZE)) {
    fprintf(stderr, "LIBXSMM: %s (munmap)\n", strerror(errno));
  }
#endif

  if (0 <= fd) {
    close(fd);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    fprintf(stderr, "LIBXSMM: invalid file descriptor (%i)\n", fd);
  }
#endif
}
#endif /*!defined(_WIN32) && !defined(__CYGWIN__)*/

LIBXSMM_RETARGETABLE int libxsmm_trace_maxdepth = -1;
LIBXSMM_RETARGETABLE int libxsmm_trace_threadid = -1;


#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_trace_init(int filter_maxdepth, int filter_threadid)
{
  int result;
#if defined(_WIN32) || defined(__CYGWIN__)
  SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
  result = FALSE != SymInitialize(GetCurrentProcess(), NULL, TRUE)
    ? EXIT_SUCCESS
    : GetLastError();
#else
  result = pthread_key_create(&libxsmm_trace_key, internal_delete);
#endif
  libxsmm_trace_maxdepth = filter_maxdepth;
  libxsmm_trace_threadid = filter_threadid;
#if defined(LIBXSMM_TRACE_STDATOMIC)
  __atomic_store_n(&libxsmm_trace_initialized, -1, __ATOMIC_SEQ_CST);
#else
  libxsmm_trace_initialized = -1;
#endif
  return result;
}


#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_trace_finalize(void)
{
  int result;
#if defined(LIBXSMM_TRACE_STDATOMIC)
  __atomic_store_n(&libxsmm_trace_initialized, -2, __ATOMIC_SEQ_CST);
#else
  libxsmm_trace_initialized = -2;
#endif
#if defined(_WIN32) || defined(__CYGWIN__)
  result = FALSE != SymCleanup(GetCurrentProcess())
    ? EXIT_SUCCESS
    : GetLastError();
#else
  result = pthread_key_delete(libxsmm_trace_key);
#endif
  return result;
}


#if defined(_WIN32)
/*TODO: no inline*/
#elif defined(__GNUC__)
LIBXSMM_ATTRIBUTE(noinline)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE const char* libxsmm_trace_info(unsigned int* depth, unsigned int* threadid,
  const int* filter_maxdepth, const int* filter_threadid)
{
  const int max_n = depth ? (LIBXSMM_TRACE_MAXDEPTH) : 2;
  const int min_n = depth ? (LIBXSMM_TRACE_MINDEPTH + *depth) : 2;
  void *stack[LIBXSMM_TRACE_MAXDEPTH], **symbol = stack + LIBXSMM_MIN(depth ? ((int)(*depth + 1)) : 1, max_n - 1);
  const char *fname = NULL;
  int i;

#if defined(__GNUC__)
  __asm__("");
#endif
#if defined(LIBXSMM_TRACE_STDATOMIC)
  i = __atomic_load_n(&libxsmm_trace_initialized, __ATOMIC_RELAXED);
#else
  i = libxsmm_trace_initialized;
#endif

  if (-1 <= i) { /* do nothing if not yet initialized */
    const int maxdepth = filter_maxdepth ? *filter_maxdepth : libxsmm_trace_maxdepth;
#if defined(_WIN32) || defined(__CYGWIN__)
    i = CaptureStackBackTrace(0, max_n, stack, NULL);
#else
    i = backtrace(stack, max_n);
#endif
    if (0 > maxdepth || i <= maxdepth + min_n) { /* filter against filter_maxdepth */
      if (min_n <= i) { /* check against min. depth */
#if defined(_WIN32) || defined(__CYGWIN__)
        const int filter = filter_threadid ? *filter_threadid : libxsmm_trace_threadid;
        static LIBXSMM_TLS char buffer[sizeof(SYMBOL_INFO)+LIBXSMM_TRACE_SYMBOLSIZE];
        static LIBXSMM_TLS int tid = -1;

        PSYMBOL_INFO value = (PSYMBOL_INFO)buffer;
        value->SizeOfStruct = sizeof(SYMBOL_INFO);
        value->MaxNameLen = LIBXSMM_TRACE_SYMBOLSIZE - 1;

        if (0 > tid) {
#if defined(_WIN32)
          const int counter = _InterlockedIncrement(&libxsmm_trace_initialized);
#elif defined(LIBXSMM_TRACE_STDATOMIC)
          const int counter = __atomic_add_fetch(&libxsmm_trace_initialized, 1, __ATOMIC_RELAXED);
#else
          const int counter = __sync_add_and_fetch(&libxsmm_trace_initialized, 1);
#endif
          assert(0 <= counter);
          tid = counter;
        }

        if (0 > filter || filter == tid) {
          if (FALSE != SymFromAddr(GetCurrentProcess(), (DWORD64)*symbol, NULL, value)
            && 0 < value->NameLen)
          {
            fname = value->Name;
            if (depth) *depth = i - min_n;
            if (threadid) *threadid = tid;
          }
# if !defined(NDEBUG) /* library code is expected to be mute */
          else {
            fprintf(stderr, "LIBXSMM: failed to translate symbol (%p)\n", *symbol);
          }
# endif
        }
#else
        char* value = (char*)pthread_getspecific(libxsmm_trace_key);
        int* ivalue = 0, fd = -1;

        if (value) {
          const int filter = filter_threadid ? *filter_threadid : libxsmm_trace_threadid;
          ivalue = (int*)value;

          if (0 > filter || filter == ivalue[1]/*tid*/) {
            fd = ivalue[0];
            if (0 <= fd && (sizeof(int) * 2) == lseek(fd, sizeof(int) * 2, SEEK_SET)) {
              value += sizeof(int) * 2;
            }
# if !defined(NDEBUG) /* library code is expected to be mute */
            else {
              fprintf(stderr, "LIBXSMM: failed to get buffer\n");
            }
# endif
          }
        }
        else {
          char filename[] = "/tmp/fileXXXXXX";
          fd = mkstemp(filename);

          if (0 <= fd && 0 == posix_fallocate(fd, 0, LIBXSMM_TRACE_SYMBOLSIZE)) {
            char *const buffer = (char*)mmap(NULL, LIBXSMM_TRACE_SYMBOLSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

            if (MAP_FAILED != buffer) {
              int check = -1;
              ivalue = (int*)buffer;
              ivalue[0] = fd; /* valid fd for internal_delete */

              if (0 == pthread_setspecific(libxsmm_trace_key, buffer)
                && sizeof(int)*1 == read(fd, &check, sizeof(int))
                && sizeof(int)*2 == lseek(fd, sizeof(int), SEEK_CUR)
                && check == fd)
              {
# if defined(LIBXSMM_TRACE_STDATOMIC)
                const int counter = __atomic_add_fetch(&libxsmm_trace_initialized, 1, __ATOMIC_RELAXED);
# else
                const int counter = __sync_add_and_fetch(&libxsmm_trace_initialized, 1);
# endif
                value = buffer + sizeof(int) * 2;
                ivalue[1] = counter;
              }
              else {
# if !defined(NDEBUG) /* library code is expected to be mute */
                fprintf(stderr, "LIBXSMM: failed to setup buffer\n");
# endif
                internal_delete(buffer);
              }
            }
# if !defined(NDEBUG)
            else {
              fprintf(stderr, "LIBXSMM: %s (mmap)\n", strerror(errno));
            }
# endif
          }
# if !defined(NDEBUG) /* library code is expected to be mute */
          else {
            fprintf(stderr, "LIBXSMM: failed to setup file descriptor (%i)\n", fd);
          }
# endif
        }

        if (value) {
          backtrace_symbols_fd(symbol, 1, fd);
          if (1 == sscanf(value, "%*[^(](%s0x", value)) {
            char* c;
            for (c = value; '+' != *c && *c; ++c);
            if ('+' == *c) {
              assert(ivalue);
              if (depth) *depth = i - min_n;
              if (threadid) *threadid = ivalue[1];
              fname = value;
              *c = 0;
            }
          }
        }
#endif
      }
# if !defined(NDEBUG) /* library code is expected to be mute */
      else {
        fprintf(stderr, "LIBXSMM: failed to capture stack trace\n");
      }
# endif
    }
  }

  return fname;
}


#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_trace(FILE* stream, unsigned int depth,
  const int* filter_maxdepth, const int* filter_threadid)
{
  unsigned int depth1 = depth + 1, threadid;
  const char *const name = libxsmm_trace_info(&depth1, &threadid,
    filter_maxdepth, filter_threadid);

  if (name && *name) { /* implies actual other results to be valid */
    assert(0 != stream/*otherwise fprintf handle the error*/);
    if ((0 == filter_threadid && 0 > libxsmm_trace_threadid) || (filter_threadid && 0 > *filter_threadid)) {
      fprintf(stream, "%*s%s@%u\n", depth1, "", name, threadid);
    }
    else {
      fprintf(stream, "%*s%s\n", depth1, "", name);
    }
  }
}


#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void __cyg_profile_func_enter(void* this_fn, void* call_site)
{
  LIBXSMM_UNUSED(this_fn); LIBXSMM_UNUSED(call_site); /* suppress warning */
  libxsmm_trace(stderr, 1/*no need for parent (0) but parent of parent (1)*/,
    /* inherit global settings from libxsmm_trace_init */
    NULL, NULL);
}


LIBXSMM_ATTRIBUTE(no_instrument_function)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void __cyg_profile_func_exit(void* this_fn, void* call_site)
{
  LIBXSMM_UNUSED(this_fn); LIBXSMM_UNUSED(call_site); /* suppress warning */
}
#endif /*defined(__GNUC__)*/

