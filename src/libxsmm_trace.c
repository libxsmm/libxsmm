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
#include <libxsmm_sync.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#include <stdlib.h>
# if (!defined(_BSD_SOURCE) || 0 == _BSD_SOURCE) && (!defined(_SVID_SOURCE) || 0 == _SVID_SOURCE) && \
     (!defined(_XOPEN_SOURCE) || !defined(_XOPEN_SOURCE_EXTENDED) || 500 > _XOPEN_SOURCE) && \
     (!defined(_POSIX_C_SOURCE) || 200112L > _POSIX_C_SOURCE)
/* C89: avoid warning about mkstemp declared implicitly */
int mkstemp(char* filename_template);
# endif
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
#else
# include <execinfo.h>
# include <sys/stat.h>
# include <sys/mman.h>
# include <unistd.h>
# include <fcntl.h>
# if defined(__APPLE__) && defined(__MACH__)
/* taken from "libtransmission" fdlimit.c */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE int posix_fallocate(int fd, off_t offset, off_t length)
{
  fstore_t fst;
  fst.fst_flags = F_ALLOCATECONTIG;
  fst.fst_posmode = F_PEOFPOSMODE;
  fst.fst_offset = offset;
  fst.fst_length = length;
  fst.fst_bytesalloc = 0;
  return fcntl(fd, F_PREALLOCATE, &fst);
}
# elif (!defined(_XOPEN_SOURCE) || 600 > _XOPEN_SOURCE) && \
       (!defined(_POSIX_C_SOURCE) || 200112L > _POSIX_C_SOURCE)
/* C89: avoid warning about posix_fallocate declared implicitly */
int posix_fallocate(int, off_t, off_t);
# endif
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
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

#if defined(_WIN32) || defined(__CYGWIN__)
# define LIBXSMM_TRACE_MINDEPTH 5
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE volatile LONG internal_trace_initialized /*= -1*/;
#else
# define LIBXSMM_TRACE_MINDEPTH 4
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int internal_trace_initialized /*= -1*/;
#if !defined(LIBXSMM_NO_SYNC)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE pthread_key_t internal_trace_key /*= 0*/;
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
    const int error = errno;
    fprintf(stderr, "LIBXSMM: %s (munmap error #%i at %p)\n",
      strerror(error), error, value);
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


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int internal_trace_mindepth /*=  0*/;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int internal_trace_threadid /*= -1*/;
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int internal_trace_maxnsyms /*= -1*/;


LIBXSMM_API
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
int libxsmm_trace_init(int filter_threadid, int filter_mindepth, int filter_maxnsyms);

LIBXSMM_API_DEFINITION int libxsmm_trace_init(int filter_threadid, int filter_mindepth, int filter_maxnsyms)
{
  int result = EXIT_SUCCESS;
  internal_trace_initialized = -1; /* disabled */
#if defined(__TRACE)
  if (0 != filter_maxnsyms) { /* enabled */
# if defined(_WIN32) || defined(__CYGWIN__)
    SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
    result = (FALSE != SymInitialize(GetCurrentProcess(), NULL, TRUE) ? EXIT_SUCCESS : GetLastError());
# elif !defined(LIBXSMM_NO_SYNC)
    result = pthread_key_create(&internal_trace_key, internal_delete);
# endif
    if (EXIT_SUCCESS == result) {
      internal_trace_threadid = filter_threadid;
      internal_trace_maxnsyms = filter_maxnsyms;
      internal_trace_mindepth = filter_mindepth;
      internal_trace_initialized = 0; /* enabled */
    }
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

LIBXSMM_API_DEFINITION int libxsmm_trace_finalize(void)
{
  int result;
#if !defined(__TRACE)
  result = EXIT_FAILURE;
#else
  const int initialized = LIBXSMM_ATOMIC_LOAD(&internal_trace_initialized, LIBXSMM_ATOMIC_RELAXED);
  if (0 == initialized) {
    LIBXSMM_ATOMIC_STORE(&internal_trace_initialized, -1/*disable*/, LIBXSMM_ATOMIC_SEQ_CST);
# if defined(_WIN32) || defined(__CYGWIN__)
    result = (FALSE != SymCleanup(GetCurrentProcess()) ? EXIT_SUCCESS : GetLastError());
# elif !defined(LIBXSMM_NO_SYNC)
    result = pthread_key_delete(internal_trace_key);
# endif
  }
  else {
    result = EXIT_SUCCESS;
  }
#endif
  return result;
}


LIBXSMM_API
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
const char* libxsmm_trace_info(unsigned int* depth, unsigned int* threadid, const int* filter_threadid, const int* filter_mindepth, const int* filter_maxnsyms);

LIBXSMM_API_DEFINITION
#if defined(_WIN32)
/*TODO: no inline*/
#elif defined(__GNUC__)
/*LIBXSMM_ATTRIBUTE(noinline)*/
#endif
const char* libxsmm_trace_info(unsigned int* depth, unsigned int* threadid, const int* filter_threadid, const int* filter_mindepth, const int* filter_maxnsyms)
{
  const char *fname = NULL;
#if defined(__TRACE)
  const int max_n = depth ? (LIBXSMM_TRACE_MAXDEPTH) : 2;
  const int min_n = depth ? (LIBXSMM_TRACE_MINDEPTH + *depth) : 2;
  void *stack[LIBXSMM_TRACE_MAXDEPTH], **symbol = stack + LIBXSMM_MIN(depth ? ((int)(*depth + 1)) : 1, max_n - 1);
  static LIBXSMM_TLS int cerberus = 0;
  int i;

  /* check against entering a recursion (recursion should not happen due to
   * attribute "no_instrument_function" but better prevent this in any case)
   */
  if (0 == cerberus) {
    ++cerberus;
# if defined(__GNUC__)
    __asm__("");
# endif
    i = LIBXSMM_ATOMIC_LOAD(&internal_trace_initialized, LIBXSMM_ATOMIC_RELAXED);
    if (0 <= i) { /* do nothing if not yet initialized */
      const int mindepth = filter_mindepth ? *filter_mindepth : internal_trace_mindepth;
      const int maxnsyms = filter_maxnsyms ? *filter_maxnsyms : internal_trace_maxnsyms;
# if defined(_WIN32) || defined(__CYGWIN__)
      i = CaptureStackBackTrace(0, max_n, stack, NULL);
# else
      i = backtrace(stack, max_n);
# endif
      /* filter depth against filter_mindepth and filter_maxnsyms */
      if ((0 >= mindepth ||      (min_n + mindepth) <= i) &&
          (0 >  maxnsyms || i <= (min_n + mindepth + maxnsyms - 1)))
      {
        if (min_n <= i) { /* check against min. depth */
          const int filter = (filter_threadid ? *filter_threadid : internal_trace_threadid);
          int abs_tid = 0;
# if defined(_WIN32) || defined(__CYGWIN__)
          static LIBXSMM_TLS char buffer[sizeof(SYMBOL_INFO)+LIBXSMM_TRACE_SYMBOLSIZE];
          static LIBXSMM_TLS int tid = 0;

          PSYMBOL_INFO value = (PSYMBOL_INFO)buffer;
          value->SizeOfStruct = sizeof(SYMBOL_INFO);
          value->MaxNameLen = LIBXSMM_TRACE_SYMBOLSIZE - 1;

          if (0 != tid) {
            abs_tid = (0 <= tid ? tid : -tid);
          }
          else {
            abs_tid = LIBXSMM_ATOMIC_ADD_FETCH(&internal_trace_initialized, 1, LIBXSMM_ATOMIC_RELAXED);
            /* use sign bit to flag enabled fallback for symbol resolution */
            tid = -abs_tid;
          }

          assert(0 < abs_tid);
          if (0 > filter || filter == abs_tid - 1) {
            if (FALSE != SymFromAddr(GetCurrentProcess(), (DWORD64)*symbol, NULL, value)
              && 0 < value->NameLen)
            {
              /* disable fallback allowing unresolved symbol names */
              tid = abs_tid; /* make unsigned */
              fname = value->Name;
            }
            else if (0 > tid) { /* fallback allowing unresolved symbol names */
              sprintf(buffer, "0x%llx", (unsigned long long)*symbol);
              fname = buffer;
            }
            if (depth) *depth = i - min_n;
            if (threadid) *threadid = abs_tid - 1;
          }
# else
#   if defined(LIBXSMM_NO_SYNC)
          static char raw_c;
          char *const raw_value = &raw_c;
#   else
          char *const raw_value = (char*)pthread_getspecific(internal_trace_key);
#   endif
          int* ivalue = 0, fd = -1;
          char* value = 0;

          if (raw_value) {
            ivalue = (int*)raw_value;
            abs_tid = (0 <= ivalue[1] ? ivalue[1] : -ivalue[1]);

            if (0 > filter || filter == abs_tid - 1) {
              fd = ivalue[0];
              if (0 <= fd && (sizeof(int) * 2) == lseek(fd, sizeof(int) * 2, SEEK_SET)) {
                value = raw_value + sizeof(int) * 2;
              }
#   if !defined(NDEBUG) /* library code is expected to be mute */
              else {
                fprintf(stderr, "LIBXSMM: failed to get buffer\n");
              }
#   endif
            }
          }
          else {
            char filename[] = "/tmp/fileXXXXXX";
            fd = mkstemp(filename);

            if (0 <= fd && 0 == posix_fallocate(fd, 0, LIBXSMM_TRACE_SYMBOLSIZE)) {
              char *const buffer = (char*)mmap(NULL, LIBXSMM_TRACE_SYMBOLSIZE,
                PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

              if (MAP_FAILED != buffer) {
                int check = -1;
                ivalue = (int*)buffer;
                ivalue[0] = fd; /* valid file descriptor for internal_delete */

                if (
#   if !defined(LIBXSMM_NO_SYNC)
                  0 == pthread_setspecific(internal_trace_key, buffer) &&
#   endif
                     (sizeof(int) * 1) == read(fd, &check, sizeof(int))
                  && (sizeof(int) * 2) == lseek(fd, sizeof(int), SEEK_CUR)
                  && check == fd)
                {
                  abs_tid = LIBXSMM_ATOMIC_ADD_FETCH(&internal_trace_initialized, 1, LIBXSMM_ATOMIC_RELAXED);
                  assert(0 < abs_tid);
                  /* use sign bit to flag enabled fallback for symbol resolution */
                  ivalue[1] = -abs_tid;

                  if (0 > filter || filter == abs_tid - 1) {
                    value = buffer + sizeof(int) * 2;
                  }
                }
                else {
#   if !defined(NDEBUG) /* library code is expected to be mute */
                  fprintf(stderr, "LIBXSMM: failed to setup buffer\n");
#   endif
                  internal_delete(buffer);
                }
              }
#   if !defined(NDEBUG)
              else {
                const int error = errno;
                fprintf(stderr, "LIBXSMM: %s (mmap allocation error #%i)\n",
                  strerror(error), error);
              }
#   endif
            }
#   if !defined(NDEBUG) /* library code is expected to be mute */
            else {
              fprintf(stderr, "LIBXSMM: failed to setup file descriptor (%i)\n", fd);
            }
#   endif
          }

          if (value) {
            backtrace_symbols_fd(symbol, 1, fd);

            /* attempt to parse symbol name */
            if (1 == sscanf(value, "%*[^(](%s0x", value)) {
              char* c;
              for (c = value; '+' != *c && *c; ++c);
              if ('+' == *c) {
                /* disable fallback allowing unresolved symbol names */
                ivalue[1] = abs_tid; /* make unsigned */
                fname = value;
                *c = 0;
              }
            }

            /* fallback to symbol address */
            if (0 > ivalue[1] && 0 == fname) {
              sprintf(value, "0x%llx", (unsigned long long)*symbol);
              fname = value;
            }

            if (depth) *depth = i - min_n;
            if (threadid) *threadid = abs_tid - 1;
          }
# endif
        }
      }
    }

    --cerberus;
  }
#else
  LIBXSMM_UNUSED(depth); LIBXSMM_UNUSED(threadid);
  LIBXSMM_UNUSED(filter_threadid);
  LIBXSMM_UNUSED(filter_mindepth);
  LIBXSMM_UNUSED(filter_maxnsyms);
#endif /*defined(__TRACE)*/

  return fname;
}


LIBXSMM_API
#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void libxsmm_trace(FILE* stream, unsigned int depth, const int* filter_threadid, const int* filter_mindepth, const int* filter_maxnsyms);

LIBXSMM_API_DEFINITION void libxsmm_trace(FILE* stream, unsigned int depth, const int* filter_threadid, const int* filter_mindepth, const int* filter_maxnsyms)
{
#if defined(__TRACE)
  unsigned int depth1 = depth + 1, threadid;
  const char *const name = libxsmm_trace_info(&depth1, &threadid,
    filter_threadid, filter_mindepth, filter_maxnsyms);

  if (name && *name) { /* implies actual other results to be valid */
    const int depth0 = LIBXSMM_MAX(filter_mindepth ? *filter_mindepth : internal_trace_mindepth, 0);
    assert(0 != stream/*otherwise fprintf handle the error*/);
    if ((0 == filter_threadid && 0 > internal_trace_threadid) || (0 != filter_threadid && 0 > *filter_threadid)) {
      fprintf(stream, "%*s%s@%u\n", (int)(depth1 - depth0), "", name, threadid);
    }
    else {
      fprintf(stream, "%*s%s\n", (int)(depth1 - depth0), "", name);
    }
  }
#else /* suppress warning */
  LIBXSMM_UNUSED(stream); LIBXSMM_UNUSED(depth);
  LIBXSMM_UNUSED(filter_threadid);
  LIBXSMM_UNUSED(filter_mindepth);
  LIBXSMM_UNUSED(filter_maxnsyms);
#endif
}


#if defined(__GNUC__)

LIBXSMM_EXTERN_C
#if defined(__cplusplus) || defined(LIBXSMM_BUILD)
LIBXSMM_API
#else
static
#endif
LIBXSMM_ATTRIBUTE(no_instrument_function) void __cyg_profile_func_enter(void* this_fn, void* call_site);

LIBXSMM_EXTERN_C
#if defined(__cplusplus) || defined(LIBXSMM_BUILD)
LIBXSMM_API_DEFINITION
#else
static
#endif
void __cyg_profile_func_enter(void* this_fn, void* call_site)
{
#if defined(__TRACE)
# if 1
  LIBXSMM_UNUSED(this_fn); LIBXSMM_UNUSED(call_site); /* suppress warning */
  libxsmm_trace(stderr, 1/*no need for parent (0) but parent of parent (1)*/,
    /* inherit global settings from libxsmm_trace_init */
    NULL, NULL, NULL);
# else
  struct {
      const char *dli_fname;
      void       *dli_fbase;  /* Address at which shared object
                                 is loaded */
      const char *dli_sname;  /* Name of nearest symbol with address
                                 lower than addr */
      void       *dli_saddr;
  } info;
  if (dladdr(this_fn, &info)) {
    if (info.dli_sname) {
      fprintf(stderr, "%s\n", info.dli_sname);
    }
    else if (info.dli_saddr) {
      fprintf(stderr, "0x%llx\n", (unsigned long long)info.dli_saddr);
    }
  }
# endif
#else
  LIBXSMM_UNUSED(this_fn); LIBXSMM_UNUSED(call_site); /* suppress warning */
#endif
}


LIBXSMM_EXTERN_C
#if defined(__cplusplus) || defined(LIBXSMM_BUILD)
LIBXSMM_API
#else
static
#endif
LIBXSMM_ATTRIBUTE(no_instrument_function) void __cyg_profile_func_exit(void* this_fn, void* call_site);

LIBXSMM_EXTERN_C
#if defined(__cplusplus) || defined(LIBXSMM_BUILD)
LIBXSMM_API_DEFINITION
#else
static
#endif
void __cyg_profile_func_exit(void* this_fn, void* call_site)
{
  LIBXSMM_UNUSED(this_fn); LIBXSMM_UNUSED(call_site); /* suppress warning */
}

#endif /*defined(__GNUC__)*/

