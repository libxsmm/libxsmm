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

#if !defined(LIBXSMM_TRACE_MINDEPTH) || 0 > (LIBXSMM_TRACE_MINDEPTH)
# undef LIBXSMM_TRACE_MINDEPTH
# define LIBXSMM_TRACE_MINDEPTH 1
#endif
#if !defined(LIBXSMM_TRACE_MAXDEPTH) || 0 >= (LIBXSMM_TRACE_MAXDEPTH)
# undef LIBXSMM_TRACE_MAXDEPTH
# define LIBXSMM_TRACE_MAXDEPTH 1024
#endif
#if !defined(LIBXSMM_TRACE_SYMBOLSIZE) || 0 >= (LIBXSMM_TRACE_SYMBOLSIZE)
# undef LIBXSMM_TRACE_SYMBOLSIZE
# define LIBXSMM_TRACE_SYMBOLSIZE 256
#endif
#if !defined(LIBXSMM_TRACE_DLINFO) && defined(__USE_GNU)
# define LIBXSMM_TRACE_DLINFO
#endif

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
#   pragma comment(lib, "dbghelp")
# endif
# if defined(_MSC_VER)
#   pragma warning(pop)
# endif
LIBXSMM_APIVAR_DEFINE(volatile LONG internal_trace_initialized);
#else
LIBXSMM_APIVAR_DEFINE(volatile int internal_trace_initialized);
# include <execinfo.h>
# if defined(LIBXSMM_TRACE_DLINFO)
#   include <dlfcn.h>
# else
#   include <sys/stat.h>
#   include <sys/mman.h>
#   include <fcntl.h>
#   if (0 != LIBXSMM_SYNC)
LIBXSMM_APIVAR_DEFINE(LIBXSMM_TLS_TYPE internal_trace_key);
LIBXSMM_APIVAR_DEFINE(void* internal_trace_symbols[LIBXSMM_NTHREADS_MAX]);
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
/*#   elif (!defined(_XOPEN_SOURCE) || 600 > _XOPEN_SOURCE) && \
         (!defined(_POSIX_C_SOURCE) || 200112L > _POSIX_C_SOURCE)*/
/* C89: avoid warning about posix_fallocate declared implicitly */
#   else
LIBXSMM_EXTERN int posix_fallocate(int, off_t, off_t);
#   endif
# endif
#endif

LIBXSMM_APIVAR_DEFINE(int internal_trace_mindepth);
LIBXSMM_APIVAR_DEFINE(int internal_trace_threadid);
LIBXSMM_APIVAR_DEFINE(int internal_trace_maxnsyms);


LIBXSMM_API LIBXSMM_ATTRIBUTE_NO_TRACE int libxsmm_trace_init(int /*filter_threadid*/, int /*filter_mindepth*/, int /*filter_maxnsyms*/);
LIBXSMM_API int libxsmm_trace_init(int filter_threadid, int filter_mindepth, int filter_maxnsyms)
{
  int result = EXIT_SUCCESS;
  if (0 == internal_trace_initialized) {
    if (0 <= filter_threadid) ++filter_threadid;
#if defined(__TRACE)
    { const char *const env = getenv("LIBXSMM_TRACE");
      if (NULL != env && 0 != *env) {
        char buffer[32] = { 0 };
        if (1 == sscanf(env, "%31[^,],", buffer)) {
          result = (0 <= sscanf(buffer, "%i", &filter_threadid) ? EXIT_SUCCESS : EXIT_FAILURE);
        }
        if (1 == sscanf(env, "%*[^,],%31[^,],", buffer)) {
          result = (0 <= sscanf(buffer, "%i", &filter_mindepth) ? EXIT_SUCCESS : EXIT_FAILURE);
        }
        if (1 == sscanf(env, "%*[^,],%*[^,],%31s", buffer)) {
          result = (0 <= sscanf(buffer, "%i", &filter_maxnsyms) ? EXIT_SUCCESS : EXIT_FAILURE);
        }
        else {
          filter_maxnsyms = -1; /* all */
        }
        if (EXIT_SUCCESS == result) {
          internal_trace_initialized = -1; /* auto */
        }
      }
    }
    if (EXIT_SUCCESS == result)
#endif
    {
#if defined(LIBXSMM_TRACE)
# if defined(_WIN32) || defined(__CYGWIN__)
      SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
      result = (FALSE != SymInitialize(GetCurrentProcess(), NULL, TRUE) ? EXIT_SUCCESS : GetLastError());
# elif (0 != LIBXSMM_SYNC) && !defined(LIBXSMM_TRACE_DLINFO)
      result = LIBXSMM_TLS_CREATE(&internal_trace_key);
# endif
      if (EXIT_SUCCESS == result) {
        internal_trace_threadid = filter_threadid;
        internal_trace_maxnsyms = filter_maxnsyms;
        internal_trace_mindepth = filter_mindepth;
        if (0 == internal_trace_initialized) {
          internal_trace_initialized = 1;
        }
      }
#else
      LIBXSMM_UNUSED(filter_threadid);
      LIBXSMM_UNUSED(filter_mindepth);
      LIBXSMM_UNUSED(filter_maxnsyms);
#endif
    }
  }
  return result;
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_NO_TRACE int libxsmm_trace_finalize(void);
LIBXSMM_API int libxsmm_trace_finalize(void)
{
  int result;
#if defined(LIBXSMM_TRACE)
  result = EXIT_SUCCESS;
  if (0 != internal_trace_initialized) {
    internal_trace_initialized = 0; /* disable */
# if defined(_WIN32) || defined(__CYGWIN__)
    result = (FALSE != SymCleanup(GetCurrentProcess()) ? EXIT_SUCCESS : GetLastError());
# elif (0 != LIBXSMM_SYNC) && !defined(LIBXSMM_TRACE_DLINFO)
    result = LIBXSMM_TLS_DESTROY(internal_trace_key);
    { int i = 0;
      for (; i < LIBXSMM_NTHREADS_MAX; ++i) {
        void *const buffer = internal_trace_symbols[i];
        if (NULL != buffer) internal_delete(buffer);
      }
    }
# endif
  }
#else
  result = EXIT_FAILURE;
#endif
  return result;
}


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
LIBXSMM_API LIBXSMM_ATTRIBUTE_NO_TRACE unsigned int libxsmm_backtrace(const void* /*buffer*/[], unsigned int /*size*/, unsigned int /*skip*/);
LIBXSMM_API
#if defined(_WIN32)
/*LIBXSMM_ATTRIBUTE(noinline)*/
#elif defined(__GNUC__)
/*LIBXSMM_ATTRIBUTE(noinline)*/
#endif
unsigned int libxsmm_backtrace(const void* buffer[], unsigned int size, unsigned int skip)
{
  unsigned int result;
  if (NULL != buffer && 0 != size && skip < size) {
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
  }
  else {
    result = 0;
  }
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
    strncpy(map, info.dli_sname, LIBXSMM_TRACE_SYMBOLSIZE - 1);
    result = map;
  }
#else
  LIBXSMM_ASSERT(NULL != address && NULL != map);
  backtrace_symbols_fd((void**)&address, 1, fd);
  if (fdoff == lseek(fd, fdoff, SEEK_SET) /* reset map */
    /* limit input to 256 characters (LIBXSMM_TRACE_SYMBOLSIZE) */
    && 1 == sscanf(map, "%*[^(](%256s0x", map))
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
#pragma GCC diagnostic pop


LIBXSMM_API LIBXSMM_ATTRIBUTE_NO_TRACE
const char* libxsmm_trace_info(unsigned int* /*depth*/, unsigned int* /*threadid*/, const int* /*filter_threadid*/,
  const void* /*filter_symbol*/, const int* /*filter_mindepth*/, const int* /*filter_maxnsyms*/);

LIBXSMM_API
#if defined(_WIN32)
/*LIBXSMM_ATTRIBUTE(noinline)*/
#elif defined(__GNUC__)
/*LIBXSMM_ATTRIBUTE(noinline)*/
#endif
const char* libxsmm_trace_info(unsigned int* depth, unsigned int* threadid, const int* filter_threadid,
  const void* filter_symbol, const int* filter_mindepth, const int* filter_maxnsyms)
{
  const char *fname = NULL;
#if defined(LIBXSMM_TRACE)
  static LIBXSMM_TLS int cerberus = 0;
  /* check against entering a recursion (recursion should not happen due to
   * attribute "no_instrument_function" but better prevent this in any case)
   */
  if (0 == cerberus) {
    int init;
    ++cerberus;
# if defined(__GNUC__) && !defined(_CRAYC)
    __asm__("");
# endif
    init = LIBXSMM_ATOMIC_LOAD(&internal_trace_initialized, LIBXSMM_ATOMIC_RELAXED);
    if (0 != init) { /* do nothing if not yet initialized */
      const int mindepth = (NULL != filter_mindepth ? *filter_mindepth : internal_trace_mindepth);
      const int maxnsyms = (NULL != filter_maxnsyms ? *filter_maxnsyms : internal_trace_maxnsyms);
      const void *stacktrace[LIBXSMM_TRACE_MAXDEPTH];
      const int n = libxsmm_backtrace(stacktrace, LIBXSMM_TRACE_MAXDEPTH, 0);
      if (0 < n) {
        const int filter = (NULL != filter_threadid ? *filter_threadid : internal_trace_threadid);
        int symbol = 0, abs_tid = 0;
# if defined(_WIN32) || defined(__CYGWIN__) || defined(LIBXSMM_TRACE_DLINFO)
        static LIBXSMM_TLS struct {
#   if defined(_WIN32) || defined(__CYGWIN__)
          char buffer[sizeof(SYMBOL_INFO)+LIBXSMM_TRACE_SYMBOLSIZE];
#   else
          char buffer[LIBXSMM_TRACE_SYMBOLSIZE];
#   endif
          int tid;
        } info /*= { 0 }*/;
        if (0 != info.tid) {
          abs_tid = LIBXSMM_ABS(info.tid);
        }
        else {
          const int tid = LIBXSMM_ATOMIC_ADD_FETCH(&internal_trace_initialized, 0 < init ? 1 : -1, LIBXSMM_ATOMIC_RELAXED);
          abs_tid = LIBXSMM_ABS(tid) - 1;
          /* use sign bit to flag enabled fallback for symbol resolution */
          info.tid = -abs_tid;
        }
        LIBXSMM_ASSERT(0 < abs_tid);
        if (0 > filter || filter == abs_tid) {
          int next = symbol + 1;
#   if defined(_WIN32) || defined(__CYGWIN__)
          const HANDLE process = GetCurrentProcess();
          PSYMBOL_INFO value = (PSYMBOL_INFO)info.buffer;
          value->SizeOfStruct = sizeof(SYMBOL_INFO);
          value->MaxNameLen = LIBXSMM_TRACE_SYMBOLSIZE - 1;
          value->NameLen = 0;
#   endif
          if (NULL != filter_symbol && '\0' != *(const char*)filter_symbol) {
            struct { size_t d; int s; } approx = { (size_t)LIBXSMM_UNLIMITED, 0 };
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
              { const size_t d = LIBXSMM_DELTA((const char*)filter_symbol, (const char*)stacktrace[symbol]);
                if (d < approx.d) {
                  approx.s = symbol + 1;
                  approx.d = d;
                }
              }
              symbol = next++;
            }
            symbol = LIBXSMM_MAX((next != n ? symbol : approx.s/*not found*/) + mindepth/*shift*/, 0);
          }
          /* apply filters based on absolute symbol position */
          if (((NULL != filter_symbol && '\0' != *(const char*)filter_symbol)
            || LIBXSMM_MAX(mindepth, 0) <= symbol) && (0 >= maxnsyms || symbol < maxnsyms))
          {
            if (symbol != next && symbol < n && filter_symbol != stacktrace[symbol] &&
#   if defined(_WIN32) || defined(__CYGWIN__)
              FALSE != SymFromAddr(process, (DWORD64)stacktrace[symbol], NULL, value) && 0 < value->NameLen)
#   else
              NULL != internal_trace_get_symbolname(stacktrace[symbol], info.buffer, 0, 0))
#   endif
            {
              /* disable fallback allowing unresolved symbol names */
              info.tid = abs_tid; /* make unsigned */
#   if defined(_WIN32) || defined(__CYGWIN__)
              fname = value->Name;
#   else
              fname = info.buffer;
#   endif
            }
            if (NULL == fname && 0 > info.tid) { /* fallback allowing unresolved symbol names */
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
        char *const raw_value = (char*)LIBXSMM_TLS_GETVALUE(internal_trace_key);
#   endif
        const off_t fdoff = sizeof(int) * 2;
        int* ivalue = NULL, fd = -1;
        char* value = NULL;
        if (NULL != raw_value) {
          ivalue = (int*)raw_value;
          abs_tid = (0 <= ivalue[1] ? ivalue[1] : -ivalue[1]);
          if (0 > filter || filter == abs_tid) {
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
          /* coverity[secure_temp] */
          fd = LIBXSMM_MKTEMP(filename);
          if (0 <= fd) {
            if (0 == unlink(filename) && 0 == posix_fallocate(fd, 0, LIBXSMM_TRACE_SYMBOLSIZE)) {
              char *const buffer = (char*)mmap(NULL, LIBXSMM_TRACE_SYMBOLSIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
              if (MAP_FAILED != buffer) {
                int check = -1;
                ivalue = (int*)buffer;
                ivalue[0] = fd; /* valid file descriptor for internal_delete */
                if (
#   if (0 != LIBXSMM_SYNC)
                  0 == LIBXSMM_TLS_SETVALUE(internal_trace_key, buffer) &&
#   endif
                  (sizeof(int) * 1) == read(fd, &check, sizeof(int)) &&
                  fdoff == lseek(fd, sizeof(int), SEEK_CUR) &&
                  check == fd)
                {
                  const int tid = LIBXSMM_ATOMIC_ADD_FETCH(&internal_trace_initialized, 0 < init ? 1 : -1, LIBXSMM_ATOMIC_RELAXED);
                  abs_tid = LIBXSMM_ABS(tid) - 1;
                  LIBXSMM_ASSERT(0 < abs_tid);
#   if (0 != LIBXSMM_SYNC)
                  LIBXSMM_ASSERT(abs_tid < LIBXSMM_NTHREADS_MAX);
                  internal_trace_symbols[abs_tid] = buffer;
#   endif
                  /* use sign bit to flag enabled fallback for symbol resolution */
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
          if (NULL != filter_symbol && '\0' != *(const char*)filter_symbol) {
            struct { size_t d; int s; } approx = { (size_t)LIBXSMM_UNLIMITED, 0 };
            while (next < n && (filter_symbol == stacktrace[symbol] ||
              NULL != internal_trace_get_symbolname(stacktrace[symbol], value, fd, fdoff)))
            {
              if (filter_symbol == stacktrace[symbol] || NULL != strstr(value, (const char*)filter_symbol)) {
                symbol = next++; /* determine the symbol after the match which is checked below */
                break;
              }
              { const size_t d = LIBXSMM_DELTA((const char*)filter_symbol, (const char*)stacktrace[symbol]);
                if (d < approx.d) {
                  approx.s = symbol + 1;
                  approx.d = d;
                }
              }
              symbol = next++;
            }
            symbol = LIBXSMM_MAX((next != n ? symbol : approx.s/*not found*/) + mindepth/*shift*/, 0);
          }
          /* apply filters based on absolute symbol position */
          if (((NULL != filter_symbol && '\0' != *(const char*)filter_symbol)
            || LIBXSMM_MAX(mindepth, 0) <= symbol) && (0 >= maxnsyms || symbol < maxnsyms))
          {
            if (symbol != next && symbol < n && filter_symbol != stacktrace[symbol] &&
              NULL != internal_trace_get_symbolname(stacktrace[symbol], value, fd, fdoff))
            {
              /* disable fallback allowing unresolved symbol names */
              ivalue[1] = abs_tid; /* make unsigned */
              fname = value;
            }
            if (NULL == fname && 0 > ivalue[1]) { /* fallback to symbol address */
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


LIBXSMM_API LIBXSMM_ATTRIBUTE_NO_TRACE
void libxsmm_trace(FILE* stream, const int* /*filter_threadid*/, const void* /*filter_symbol*/, const int* /*filter_mindepth*/, const int* /*filter_maxnsyms*/);

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

LIBXSMM_API LIBXSMM_ATTRIBUTE_NO_TRACE void __cyg_profile_func_enter(void* /*this_fn*/, void* /*call_site*/);
LIBXSMM_API void __cyg_profile_func_enter(void* this_fn, void* call_site)
{
#if defined(LIBXSMM_TRACE)
  if (0 > internal_trace_initialized) {
    /* NULL: inherit global settings from libxsmm_trace_init */
    libxsmm_trace(stderr, NULL/*filter_threadid*/, "__cyg_profile_func_enter"/*LIBXSMM_FUNCNAME*/, NULL, NULL);
  }
#endif
  LIBXSMM_UNUSED(this_fn); LIBXSMM_UNUSED(call_site);
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_NO_TRACE void __cyg_profile_func_exit(void* /*this_fn*/, void* /*call_site*/);
LIBXSMM_API void __cyg_profile_func_exit(void* this_fn, void* call_site)
{
  LIBXSMM_UNUSED(this_fn); LIBXSMM_UNUSED(call_site); /* suppress warning */
}

#endif /*defined(__TRACE) && defined(__GNUC__) && defined(LIBXSMM_BUILD)*/
