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
#ifndef LIBXSMM_TRACE_H
#define LIBXSMM_TRACE_H

#include <libxsmm_macros.h>

#if defined(__TRACE) && defined(LIBXSMM_BUILD)
# define LIBXSMM_TRACE
#endif
#if !defined(LIBXSMM_TRACE_CALLERID_MAXDEPTH)
# define LIBXSMM_TRACE_CALLERID_MAXDEPTH 8
#endif
#if !defined(LIBXSMM_TRACE_CALLERID_GCCBUILTIN) && \
  ((!defined(_WIN32) || defined(__MINGW32__) || (defined(_MSC_VER) && defined(__clang__))) && \
   (!defined(__PGI) || LIBXSMM_VERSION2(19, 0) <= LIBXSMM_VERSION2(__PGIC__, __PGIC_MINOR__)) && \
    (defined(__GNUC__) || defined(__clang__)))
# define LIBXSMM_TRACE_CALLERID_GCCBUILTIN
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

#if defined(LIBXSMM_TRACE_CALLERID_GCCBUILTIN) && defined(LIBXSMM_PRAGMA_DIAG)
# if defined(__clang__)
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wunknown-warning-option"
#   if LIBXSMM_VERSION2(9, 0) <= LIBXSMM_VERSION2(__clang_major__, __clang_minor__)
#     pragma clang diagnostic ignored "-Wframe-address"
#   endif
# elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wpragmas"
#   pragma GCC diagnostic ignored "-Wframe-address"
# endif
#endif
LIBXSMM_API_INLINE const void* libxsmm_trace_caller_id(unsigned int level) { /* must be inline */
#if defined(LIBXSMM_TRACE_CALLERID_GCCBUILTIN)
  switch (level) {
# if 0
  case 0: return __builtin_extract_return_addr(__builtin_return_address(0));
  case 1: return __builtin_extract_return_addr(__builtin_return_address(1));
  case 2: return __builtin_extract_return_addr(__builtin_return_address(2));
  case 3: return __builtin_extract_return_addr(__builtin_return_address(3));
# else
  case 0: return __builtin_frame_address(1);
  case 1: return __builtin_frame_address(2);
  case 2: return __builtin_frame_address(3);
  case 3: return __builtin_frame_address(4);
# endif
  default:
#else
  {
# if defined(_WIN32)
    if (0 == level) return _AddressOfReturnAddress();
    else
# endif
#endif
    { const void* stacktrace[LIBXSMM_TRACE_CALLERID_MAXDEPTH];
      const unsigned int n = libxsmm_backtrace(stacktrace, LIBXSMM_TRACE_CALLERID_MAXDEPTH, 0/*skip*/);
      return (level < n ? stacktrace[level] : NULL);
    }
  }
}
#if defined(LIBXSMM_TRACE_CALLERID_GCCBUILTIN) && defined(LIBXSMM_PRAGMA_DIAG)
# if defined(__clang__)
#   pragma clang diagnostic pop
# elif defined(__GNUC__)
#   pragma GCC diagnostic pop
# endif
#endif

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
