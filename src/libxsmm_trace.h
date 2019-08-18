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
#ifndef LIBXSMM_TRACE_H
#define LIBXSMM_TRACE_H

#include <libxsmm_macros.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if (defined(__TRACE) || defined(LIBXSMM_BUILD) || !defined(_WIN32))
# define LIBXSMM_TRACE
#endif


/** Initializes the trace facility; NOT thread-safe. */
LIBXSMM_API int libxsmm_trace_init(
  /* Filter for thread id (-1: all). */
  int filter_threadid,
  /* Specify min. depth of stack trace (0: all). */
  int filter_mindepth,
  /* Specify max. depth of stack trace (-1: all). */
  int filter_maxnsyms);

/** Finalizes the trace facility; NOT thread-safe. */
LIBXSMM_API int libxsmm_trace_finalize(void);

/** Receives the backtrace of up to 'size' addresses. Returns the actual number of addresses (n <= size). */
LIBXSMM_API unsigned int libxsmm_backtrace(const void* buffer[], unsigned int size, unsigned int skip);

LIBXSMM_API_INLINE const void* libxsmm_trace_caller_id(unsigned int level) { /* must be inline */
#if defined(__GNUC__) && (!defined(__PGI) || \
  LIBXSMM_VERSION3(19, 0, 0) <= LIBXSMM_VERSION3(__PGIC__, __PGIC_MINOR__, __PGIC_PATCHLEVEL__))
  if (0 == level) return __builtin_return_address(0);
  else
#elif defined(_WIN32)
  if (0 == level) return _AddressOfReturnAddress();
  else
#endif
  { const void* stacktrace[4/*sufficient/maximum level*/];
    const unsigned int n = libxsmm_backtrace(stacktrace, sizeof(stacktrace) / sizeof(*stacktrace), 0/*skip*/);
    return (level < n ? stacktrace[level] : NULL);
  }
}

/** Returns the name of the function where libxsmm_trace is called from; thread-safe. */
LIBXSMM_API const char* libxsmm_trace_info(
  /* Query and output the abs. location in stacktrace (no input). */
  unsigned int* depth,
  /* Query and output the thread id (no input). */
  unsigned int* threadid,
  /* Filter for thread id (-1: all, NULL: libxsmm_trace_init). */
  const int* filter_threadid,
  /* Lookup symbol (depth argument becomes relative to symbol position). */
  const void* filter_symbol,
  /* Specify min. abs. position in stack trace (-1 or 0: all, NULL: libxsmm_trace_init). */
  const int* filter_mindepth,
  /* Specify max. depth of stack trace (-1 or 0: all, NULL: libxsmm_trace_init). */
  const int* filter_maxnsyms);

/** Prints an entry of the function where libxsmm_trace is called from (indented/hierarchical). */
LIBXSMM_API void libxsmm_trace(FILE* stream,
  /* Filter for thread id (-1: all, NULL: libxsmm_trace_init). */
  const int* filter_threadid,
  /* Lookup symbol (depth argument becomes relative to symbol position). */
  const void* filter_symbol,
  /* Specify min. absolute pos. in stack trace (-1 or 0: all, NULL: libxsmm_trace_init). */
  const int* filter_mindepth,
  /* Specify max. depth of stack trace (-1 or 0: all, NULL: libxsmm_trace_init). */
  const int* filter_maxnsyms);

#endif /*LIBXSMM_TRACE_H*/

