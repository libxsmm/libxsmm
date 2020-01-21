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
#include <stdio.h>
#include <math.h>

#if !defined(USE_NOINIT)
# define USE_NOINIT
#endif

#if !defined(MAX_NSECONDS)
# define MAX_NSECONDS 16
#endif
#if !defined(MAX_TOLPERC)
# define MAX_TOLPERC 5
#endif

#if defined(_WIN32)
# include <Windows.h>
#else
# include <unistd.h>
#endif


LIBXSMM_INLINE int timer_sleep(unsigned int seconds)
{
  int result;
#if defined(_WIN32)
  Sleep((DWORD)(1000 * seconds));
  result = EXIT_SUCCESS;
#else
  result = (0 == sleep(seconds) ? EXIT_SUCCESS : EXIT_FAILURE);
#endif
  return result;
}


int main(int argc, char* argv[])
{
  const int max_nseconds_input = (1 < argc ? atoi(argv[1]) : MAX_NSECONDS);
  const unsigned int max_nseconds = (unsigned int)LIBXSMM_UP2POT(LIBXSMM_MAX(max_nseconds_input, 1));
  const char *const env_test = getenv("TEST_TIMER");
  const int nofailure = (NULL == env_test || 0 == *env_test) ? 0 : (0 == atoi(env_test));
  double total = 0, delta = 0, d, t;
  unsigned int n = max_nseconds;
  libxsmm_timer_tickint start;
  int result;

#if !defined(USE_NOINIT)
  libxsmm_init();
#endif

  for (n >>= 1; 0 < n; n >>= 1) {
    start = libxsmm_timer_tick();
    if (EXIT_SUCCESS == timer_sleep(n)) {
      t = libxsmm_timer_duration(start, libxsmm_timer_tick());
      d = LIBXSMM_DELTA(t, (double)n);
      if (delta < d) delta = d;
      total += t;
    }
    else total += (double)n;
  }

  start = libxsmm_timer_tick();
  if (EXIT_SUCCESS == timer_sleep(1)) {
    t = libxsmm_timer_duration(start, libxsmm_timer_tick());
    d = LIBXSMM_DELTA(t, 1.0);
    if (delta < d) delta = d;
    total += t;
  }
  else total += 1.0;

  d = LIBXSMM_DELTA(total, (double)max_nseconds);
  if (delta < d) delta = d;

  result = (int)LIBXSMM_ROUND(100.0 * delta);
  if ((MAX_TOLPERC) >= result) {
    result = EXIT_SUCCESS;
  }
  else if (0 != nofailure) {
    fprintf(stderr, "delta=%i%%\n", result);
    result = EXIT_SUCCESS;
  }

  return result;
}

