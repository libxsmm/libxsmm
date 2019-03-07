/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
#include <libxsmm_sync.h>

#if !defined(LIBXSMM_TRACE_DLINFO) /*&& !defined(__STATIC)*/
# define LIBXSMM_TRACE_DLINFO
#endif
#if !defined(LIBXSMM_TRACE_MINDEPTH) || 0 > (LIBXSMM_TRACE_MINDEPTH)
# undef LIBXSMM_TRACE_MINDEPTH
# define LIBXSMM_TRACE_MINDEPTH 1
#endif
#if !defined(LIBXSMM_TRACE_MAXDEPTH) || 0 >= (LIBXSMM_TRACE_MAXDEPTH)
# undef LIBXSMM_TRACE_MAXDEPTH
# define LIBXSMM_TRACE_MAXDEPTH 32768
#endif
#if !defined(LIBXSMM_TRACE_SYMBOLSIZE) || 0 >= (LIBXSMM_TRACE_SYMBOLSIZE)
# undef LIBXSMM_TRACE_SYMBOLSIZE
# define LIBXSMM_TRACE_SYMBOLSIZE 256
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if !defined(NDEBUG)
# include <errno.h>
#endif
#if defined(_WIN32) || defined(__CYGWIN__)
# include <Windows.h>
# if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 4091)
# endif
# include <DbgHelp.h>
# if defined(_MSC_VER)
#   pragma warning(pop)
# endif
LIBXSMM_APIVAR(volatile LONG internal_trace_initialized);
#else
LIBXSMM_APIVAR(volatile int internal_trace_initialized);
# include <execinfo.h>
# if defined(LIBXSMM_TRACE_DLINFO)
#   include <dlfcn.h>
# else
#   include <sys/stat.h>
#   include <sys/mman.h>
#   include <unistd.h>
#   include <pthread.h>
#   include <fcntl.h>
#   if (0 != LIBXSMM_SYNC)
LIBXSMM_APIVAR(pthread_key_t internal_trace_key);
#   endif
LIBXSMM_API_INLINE void internal_delete(void* value)
{
  int fd;
#   if !(defined(__APPLE__) && defined(__MACH__))
  LIBXSMM_ASSERT(NULL != value);
#   endif
  fd = *((int*)value);
#   if defined(NDEBUG)
  munmap(value, LIBXSMM_TRACE_SYMBOLSIZE);
#   else /* library code is expected to be mute */
  if (0 != munmap(value, LIBXSMM_TRACE_SYMBOLSIZE)) {
    const int error = errno;
    fprintf(stderr, "LIBXSMM ERROR: %s (munmap error #%i at %p)\n",
      strerror(error), error, value);
  }
#   endif
  if (0 <= fd) {
    close(fd);
  }
#   if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    fprintf(stderr, "LIBXSMM ERROR: invalid file descriptor (%i)\n", fd);
  }
#   endif
}
#   if defined(__APPLE__) && defined(__MACH__)
/* taken from "libtransmission" fdlimit.c */
LIBXSMM_API_INLINE int posix_fallocate(int fd, off_t offset, off_t length)
{
  fstore_t fst;
  fst.fst_flags = F_ALLOCATECONTIG;
  fst.fst_posmode = F_PEOFPOSMODE;
  fst.fst_offset = offset;
  fst.fst_length = length;
  fst.fst_bytesalloc = 0;
  return fcntl(fd, F_PREALLOCATE, &fst);
}
#   elif (!defined(_XOPEN_SOURCE) || 600 > _XOPEN_SOURCE) && \
         (!defined(_POSIX_C_SOURCE) || 200112L > _POSIX_C_SOURCE)
/* C89: avoid warning about posix_fallocate declared implicitly */
LIBXSMM_EXTERN int posix_fallocate(int, off_t, off_t);
#   endif
# endif
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_APIVAR(int internal_trace_mindepth);
LIBXSMM_APIVAR(int internal_trace_threadid);
LIBXSMM_APIVAR(int internal_trace_maxnsyms);


LIBXSMM_API
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
int libxsmm_trace_init(int filter_threadid, int filter_mindepth, int filter_maxnsyms);

LIBXSMM_API int libxsmm_trace_init(int filter_threadid, int filter_mindepth, int filter_maxnsyms)
{
  int result = EXIT_SUCCESS;
  internal_trace_initialized = -1; /* disabled */
#if defined(LIBXSMM_TRACE)
# if defined(_WIN32) || defined(__CYGWIN__)
  SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
  result = (FALSE != SymInitialize(GetCurrentProcess(), NULL, TRUE) ? EXIT_SUCCESS : GetLastError());
# elif (0 != LIBXSMM_SYNC) && !defined(LIBXSMM_TRACE_DLINFO)
  result = pthread_key_create(&internal_trace_key, internal_delete);
# endif
  if (EXIT_SUCCESS == result) {
    internal_trace_threadid = filter_threadid;
    internal_trace_maxnsyms = filter_maxnsyms;
    internal_trace_mindepth = filter_mindepth;
    internal_trace_initialized = 0; /* enabled */
  }
#else
  LIBXSMM_UNUSED(filter_threadid);
  LIBXSMM_UNUSED(filter_mindepth);
  LIBXSMM_UNUSED(filter_maxnsyms);
#endif
  return result;
}


LIBXSMM_API
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
int libxsmm_trace_finalize(void);

LIBXSMM_API int libxsmm_trace_finalize(void)
{
  int result;
#if defined(LIBXSMM_TRACE)
  result = EXIT_SUCCESS;
  if (0 <= internal_trace_initialized) {
    internal_trace_initialized = -1; /* disable */
# if defined(_WIN32) || defined(__CYGWIN__)
    result = (FALSE != SymCleanup(GetCurrentProcess()) ? EXIT_SUCCESS : GetLastError());
# elif (0 != LIBXSMM_SYNC) && !defined(LIBXSMM_TRACE_DLINFO)
    result = pthread_key_delete(internal_trace_key);
# endif
  }
#else
  result = EXIT_FAILURE;
#endif
  return result;
}


LIBXSMM_API
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
unsigned int libxsmm_backtrace(const void* buffer[], unsigned int size, unsigned int skip);

LIBXSMM_API
#if defined(_WIN32)
/*TODO: no inline*/
#elif defined(__GNUC__)
/*LIBXSMM_ATTRIBUTE(noinline)*/
#endif
unsigned int libxsmm_backtrace(const void* buffer[], unsigned int size, unsigned int skip)
{
  unsigned int result;
  skip += LIBXSMM_TRACE_MINDEPTH;
#if defined(_WIN32) || defined(__CYGWIN__)
  result = CaptureStackBackTrace(skip, LIBXSMM_MIN(size, LIBXSMM_TRACE_MAXDEPTH), (PVOID*)buffer, NULL/*hash*/);
#else
  { const int n = backtrace((void**)buffer, LIBXSMM_MIN((int)(size + skip), LIBXSMM_TRACE_MAXDEPTH));
    if ((int)skip < n) {
      result = n - skip;
      if (0 != skip) {
        memmove(buffer, buffer + skip, result * sizeof(void*));
      }
    }
    else {
      result = 0;
    }
  }
#endif
  return result;
}


#if !defined(_WIN32) && !defined(__CYGWIN__)
LIBXSMM_API_INLINE const char* internal_trace_get_symbolname(const void* address, char* map, int fd, off_t fdoff)
{
  const char* result = NULL;
#if defined(LIBXSMM_TRACE_DLINFO)
  Dl_info info;
  LIBXSMM_UNUSED(fd); LIBXSMM_UNUSED(fdoff);
  LIBXSMM_ASSERT(NULL != address && NULL != map);
  if (0 != dladdr(address, &info) && NULL != info.dli_sname) {
    strncpy(map, info.dli_sname, LIBXSMM_TRACE_SYMBOLSIZE);
    result = map;
  }
#else
  LIBXSMM_ASSERT(NULL != address && NULL != map);
  backtrace_symbols_fd((void**)&address, 1, fd);
  if (fdoff == lseek(fd, fdoff, SEEK_SET) /* reset map */
    && 1 == sscanf(map, "%*[^(](%s0x", map))
  {
    char* c = map;
    for (; '+' != *c && 0 != *c; ++c);
    if ('+' == *c && c != map) {
      result = map;
      map = c;
    }
  }
  *map = 0; /* terminate */
#endif
  return result;
}
#endif


LIBXSMM_API
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
const char* libxsmm_trace_info(unsigned int* depth, unsigned int* threadid, const int* filter_threadid,
  const void* filter_symbol, const int* filter_mindepth, const int* filter_maxnsyms);

LIBXSMM_API
#if defined(_WIN32)
/*TODO: no inline*/
#elif defined(__GNUC__)
/*LIBXSMM_ATTRIBUTE(noinline)*/
#endif
const char* libxsmm_trace_info(unsigned int* depth, unsigned int* threadid, const int* filter_threadid,
  const void* filter_symbol, const int* filter_mindepth, const int* filter_maxnsyms)
{
  const char *fname = NULL;
#if defined(LIBXSMM_TRACE)
  const void *stacktrace[LIBXSMM_TRACE_MAXDEPTH];
  static LIBXSMM_TLS int cerberus = 0;
  int n, symbol = 0;

  /* check against entering a recursion (recursion should not happen due to
   * attribute "no_instrument_function" but better prevent this in any case)
   */
  if (0 == cerberus) {
    ++cerberus;
# if defined(__GNUC__)
    __asm__("");
# endif
    n = LIBXSMM_ATOMIC_LOAD(&internal_trace_initialized, LIBXSMM_ATOMIC_RELAXED);
    if (0 <= n) { /* do nothing if not yet initialized */
      const int mindepth = (NULL != filter_mindepth ? *filter_mindepth : internal_trace_mindepth);
      const int maxnsyms = (NULL != filter_maxnsyms ? *filter_maxnsyms : internal_trace_maxnsyms);
      n = libxsmm_backtrace(stacktrace, LIBXSMM_TRACE_MAXDEPTH, 0);
      if (0 < n) {
        const int filter = (NULL != filter_threadid ? *filter_threadid : internal_trace_threadid);
        int abs_tid = 0;
# if defined(_WIN32) || defined(__CYGWIN__) || defined(LIBXSMM_TRACE_DLINFO)
        static LIBXSMM_TLS struct {
#   if defined(_WIN32) || defined(__CYGWIN__)
          char buffer[sizeof(SYMBOL_INFO)+LIBXSMM_TRACE_SYMBOLSIZE];
#   else
          char buffer[LIBXSMM_TRACE_SYMBOLSIZE];
#   endif
          int tid;
        } info;
        if (0 != info.tid) {
          abs_tid = (0 <= info.tid ? info.tid : -info.tid);
        }
        else {
          abs_tid = LIBXSMM_ATOMIC_ADD_FETCH(&internal_trace_initialized, 1, LIBXSMM_ATOMIC_RELAXED);
          /* use sign bit to flag enabled fall-back for symbol resolution */
          info.tid = -abs_tid;
        }
        LIBXSMM_ASSERT(0 < abs_tid);
        if (0 > filter || filter == abs_tid - 1) {
#   if defined(_WIN32) || defined(__CYGWIN__)
          const HANDLE process = GetCurrentProcess();
          PSYMBOL_INFO value = (PSYMBOL_INFO)info.buffer;
          value->SizeOfStruct = sizeof(SYMBOL_INFO);
          value->MaxNameLen = LIBXSMM_TRACE_SYMBOLSIZE - 1;
          value->NameLen = 0;
#   endif
          int next = symbol + 1;
          if (NULL != filter_symbol) {
            while (next < n && (filter_symbol == stacktrace[symbol] ||
#   if defined(_WIN32) || defined(__CYGWIN__)
              (FALSE != SymFromAddr(process, (DWORD64)stacktrace[symbol], NULL, value) && 0 < value->NameLen)))
            {
              if (filter_symbol == stacktrace[symbol] || NULL != strstr(value->Name, (const char*)filter_symbol)) {
#   else
              (NULL != internal_trace_get_symbolname(stacktrace[symbol], info.buffer, 0, 0))))
            {
              if (filter_symbol == stacktrace[symbol] || NULL != strstr(info.buffer, (const char*)filter_symbol)) {
#   endif
                symbol = next++; /* determine the symbol after the match which is checked below */
                break;
              }
              symbol = next++;
            }
            symbol = next != n ? LIBXSMM_CLMP(symbol + mindepth, 0, n - 1) : 0/*not found*/;
          }
          /* apply filters based on absolute symbol position */
          if ((NULL != filter_symbol || LIBXSMM_MAX(mindepth, 0) <= symbol) && (0 >= maxnsyms || symbol < maxnsyms)) {
            if (symbol != next && filter_symbol != stacktrace[symbol] &&
#   if defined(_WIN32) || defined(__CYGWIN__)
              FALSE != SymFromAddr(process, (DWORD64)stacktrace[symbol], NULL, value) && 0 < value->NameLen)
#   else
              NULL != internal_trace_get_symbolname(stacktrace[symbol], info.buffer, 0, 0))
#   endif
            {
              /* disable fall-back allowing unresolved symbol names */
              info.tid = abs_tid; /* make unsigned */
#   if defined(_WIN32) || defined(__CYGWIN__)
              fname = value->Name;
#   else
              fname = info.buffer;
#   endif
            }
            if (NULL == fname && 0 > info.tid) { /* fall-back allowing unresolved symbol names */
#   if defined(__MINGW32__)
              sprintf(info.buffer, "%p", stacktrace[symbol]);
#   else
              sprintf(info.buffer, "0x%" PRIxPTR, (uintptr_t)stacktrace[symbol]);
#   endif
              fname = info.buffer;
            }
          }
        }
# else
#   if (0 == LIBXSMM_SYNC)
        static char raw_c;
        char */*const*/ raw_value = &raw_c; /* const: avoid warning (below / constant control-flow) */
#   else
        char *const raw_value = (char*)pthread_getspecific(internal_trace_key);
#   endif
        const off_t fdoff = sizeof(int) * 2;
        int* ivalue = NULL, fd = -1;
        char* value = NULL;
        if (NULL != raw_value) {
          ivalue = (int*)raw_value;
          abs_tid = (0 <= ivalue[1] ? ivalue[1] : -ivalue[1]);

          if (0 > filter || filter == abs_tid - 1) {
            fd = ivalue[0];
            if (0 <= fd && fdoff == lseek(fd, fdoff, SEEK_SET)) {
              value = raw_value + fdoff;
            }
#   if !defined(NDEBUG) /* library code is expected to be mute */
            else {
              fprintf(stderr, "LIBXSMM ERROR: failed to get buffer\n");
            }
#   endif
          }
        }
        else {
          char filename[] = "/tmp/.libxsmm_map." LIBXSMM_MKTEMP_PATTERN;
          fd = mkstemp(filename);
          if (0 <= fd) {
            if (0 == unlink(filename) && 0 == posix_fallocate(fd, 0, LIBXSMM_TRACE_SYMBOLSIZE)) {
              char *const buffer = (char*)mmap(NULL, LIBXSMM_TRACE_SYMBOLSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
              if (MAP_FAILED != buffer) {
                int check = -1;
                ivalue = (int*)buffer;
                ivalue[0] = fd; /* valid file descriptor for internal_delete */
                if (
#   if (0 != LIBXSMM_SYNC)
                  0 == pthread_setspecific(internal_trace_key, buffer) &&
#   endif
                      (sizeof(int) * 1) == read(fd, &check, sizeof(int))
                  && fdoff == lseek(fd, sizeof(int), SEEK_CUR)
                  && check == fd)
                {
                  abs_tid = LIBXSMM_ATOMIC_ADD_FETCH(&internal_trace_initialized, 1, LIBXSMM_ATOMIC_RELAXED);
                  LIBXSMM_ASSERT(0 < abs_tid);
                  /* use sign bit to flag enabled fall-back for symbol resolution */
                  ivalue[1] = -abs_tid;
                  if (0 > filter || (abs_tid - 1) == filter) {
                    value = buffer + fdoff;
                  }
                }
                else {
#   if !defined(NDEBUG) /* library code is expected to be mute */
                  fprintf(stderr, "LIBXSMM ERROR: failed to setup buffer\n");
#   endif
                  internal_delete(buffer);
                }
              }
#   if !defined(NDEBUG)
              else {
                const int error = errno;
                fprintf(stderr, "LIBXSMM ERROR: %s (mmap allocation error #%i)\n",
                  strerror(error), error);
              }
#   endif
            }
#   if !defined(NDEBUG) /* library code is expected to be mute */
            else {
              fprintf(stderr, "LIBXSMM ERROR: failed to setup file descriptor (%i)\n", fd);
            }
#   endif
          }
        }
        if (NULL != value) {
          int next = symbol + 1;
          if (NULL != filter_symbol) {
            while (next < n && (filter_symbol == stacktrace[symbol] ||
              NULL != internal_trace_get_symbolname(stacktrace[symbol], value, fd, fdoff)))
            {
              if (filter_symbol == stacktrace[symbol] || NULL != strstr(value, (const char*)filter_symbol)) {
                symbol = next++; /* determine the symbol after the match which is checked below */
                break;
              }
              symbol = next++;
            }
            symbol = next != n ? LIBXSMM_CLMP(symbol + mindepth, 0, n - 1) : 0/*not found*/;
          }
          /* apply filters based on absolute symbol position */
          if ((NULL != filter_symbol || LIBXSMM_MAX(mindepth, 0) <= symbol) && (0 >= maxnsyms || symbol < maxnsyms)) {
            if (symbol != next && filter_symbol != stacktrace[symbol] &&
              NULL != internal_trace_get_symbolname(stacktrace[symbol], value, fd, fdoff))
            {
              /* disable fall-back allowing unresolved symbol names */
              ivalue[1] = abs_tid; /* make unsigned */
              fname = value;
            }
            if (NULL == fname && 0 > ivalue[1]) { /* fall-back to symbol address */
              sprintf(value, "0x%llx", (unsigned long long)stacktrace[symbol]);
              fname = value;
            }
          }
        }
# endif
        if (threadid) *threadid = abs_tid - 1;
        if (depth) *depth = symbol;
      }
    }
    --cerberus;
  }
#else
  LIBXSMM_UNUSED(depth);
  LIBXSMM_UNUSED(threadid);
  LIBXSMM_UNUSED(filter_threadid);
  LIBXSMM_UNUSED(filter_symbol);
  LIBXSMM_UNUSED(filter_mindepth);
  LIBXSMM_UNUSED(filter_maxnsyms);
#endif
  return fname;
}


LIBXSMM_API
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void libxsmm_trace(FILE* stream, const int* filter_threadid, const void* filter_symbol, const int* filter_mindepth, const int* filter_maxnsyms);

LIBXSMM_API void libxsmm_trace(FILE* stream, const int* filter_threadid, const void* filter_symbol, const int* filter_mindepth, const int* filter_maxnsyms)
{
#if defined(LIBXSMM_TRACE)
  unsigned int depth, threadid;
  const char *const name = libxsmm_trace_info(&depth, &threadid, filter_threadid, filter_symbol, filter_mindepth, filter_maxnsyms);
  if (NULL != name && 0 != *name) { /* implies actual other results to be valid */
    LIBXSMM_ASSERT(NULL != stream/*otherwise fprintf handles the error*/);
    if ((NULL == filter_threadid && 0 > internal_trace_threadid) || (NULL != filter_threadid && 0 > *filter_threadid)) {
      fprintf(stream, "%*s%s@%u\n", (int)depth, "", name, threadid);
    }
    else {
      fprintf(stream, "%*s%s\n", (int)depth, "", name);
    }
  }
#else /* suppress warning */
  LIBXSMM_UNUSED(stream);
  LIBXSMM_UNUSED(filter_threadid);
  LIBXSMM_UNUSED(filter_symbol);
  LIBXSMM_UNUSED(filter_mindepth);
  LIBXSMM_UNUSED(filter_maxnsyms);
#endif
}


#if defined(__TRACE) && defined(__GNUC__) && defined(LIBXSMM_BUILD)

LIBXSMM_API LIBXSMM_ATTRIBUTE(no_instrument_function) void __cyg_profile_func_enter(void* this_fn, void* call_site);
LIBXSMM_API void __cyg_profile_func_enter(void* this_fn, void* call_site)
{
#if defined(LIBXSMM_TRACE)
  /* NULL: inherit global settings from libxsmm_trace_init */
  libxsmm_trace(stderr, NULL/*filter_threadid*/, LIBXSMM_CALLER, NULL, NULL);
#endif
  LIBXSMM_UNUSED(this_fn); LIBXSMM_UNUSED(call_site);
}


LIBXSMM_API LIBXSMM_ATTRIBUTE(no_instrument_function) void __cyg_profile_func_exit(void* this_fn, void* call_site);
LIBXSMM_API void __cyg_profile_func_exit(void* this_fn, void* call_site)
{
  LIBXSMM_UNUSED(this_fn); LIBXSMM_UNUSED(call_site); /* suppress warning */
}

#endif /*defined(__TRACE) && defined(__GNUC__) && defined(LIBXSMM_BUILD)*/

