/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>

#if !defined(USE_NOINIT)
# define USE_NOINIT
#endif
#if !defined(USE_QUIET)
# define USE_QUIET
#endif
#if !defined(MAX_NSECONDS)
# define MAX_NSECONDS 16
#endif
#if !defined(MAX_TOLPERC)
# define MAX_TOLPERC 2
#endif

#if defined(_DEBUG) || !defined(USE_QUIET)
# include <stdio.h>
# define FPRINTF(STREAM, ...) fprintf(STREAM, __VA_ARGS__)
#else
# define FPRINTF(STREAM, ...)
#endif

#if defined(_WIN32)
# include <Windows.h>
# define SLEEP(SEC) Sleep(1000*(SEC))
#else
# include <unistd.h>
# define SLEEP(SEC) sleep(SEC)
#endif


int main(int argc, char* argv[])
{
  int result = EXIT_SUCCESS;
  const int max_nseconds_input = (1 < argc ? atoi(argv[1]) : MAX_NSECONDS);
  const int max_nseconds = LIBXSMM_UP2POT(max_nseconds_input);
  libxsmm_timer_tickint start;
  int n = max_nseconds;
  double total = 0, d;

#if !defined(USE_NOINIT)
  libxsmm_init();
#endif

  for (n >>= 1; 0 < n; n >>= 1) {
    start = libxsmm_timer_tick();
    SLEEP(n);
    d = libxsmm_timer_duration(start, libxsmm_timer_tick());
    total += d;
    if (((double)(MAX_TOLPERC) * n) < (100.0 * LIBXSMM_DELTA(d, (double)n))) {
      result = EXIT_FAILURE;
    }
#if defined(USE_QUIET)
    if (EXIT_SUCCESS != result)
#endif
    FPRINTF(stderr, "%i <-> %f s\n", n, d);
  }

  start = libxsmm_timer_tick();
  SLEEP(1);
  d = libxsmm_timer_duration(start, libxsmm_timer_tick());
  total += d;

  if (EXIT_SUCCESS != result /* previously exceeded tolerance */
    || ((double)(MAX_TOLPERC) * max_nseconds) < (100.0 * LIBXSMM_DELTA(total, (double)max_nseconds)))
  {
    result = EXIT_FAILURE;
  }
#if defined(USE_QUIET)
  if (EXIT_SUCCESS != result)
#endif
  FPRINTF(stderr, "%i <-> %f s\n", max_nseconds, total);

  return result;
}

