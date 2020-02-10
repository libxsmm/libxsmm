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
#if !defined(INCLUDE_LIBXSMM_LAST)
# include <libxsmm.h>
#endif
#include <math.h>
#if defined(INCLUDE_LIBXSMM_LAST)
# include <libxsmm.h>
#endif

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


libxsmm_timer_tickint libxsmm_timer_tick_rtc(void);
double libxsmm_timer_duration_rtc(libxsmm_timer_tickint, libxsmm_timer_tickint);


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
  const char *const env_delta = getenv("TIMER_DELTA");
  const int max_delta = (NULL == env_delta || 0 == *env_delta)
#if defined(_DEBUG)
    ? -1
#else
    ? 0
#endif
    : atoi(env_delta);
  libxsmm_timer_tickint begin, start, rbegin, rend;
  unsigned int n = max_nseconds, ninterrupts = 0;
  double total = 0, delta = 0, d, t;
  int result;

#if !defined(USE_NOINIT)
  libxsmm_init();
#endif

  rbegin = libxsmm_timer_tick_rtc();
  start = begin = libxsmm_timer_tick();
  for (n >>= 1; 0 < n; n >>= 1) {
    if (EXIT_SUCCESS == timer_sleep(n)) {
      t = libxsmm_timer_duration(start, libxsmm_timer_tick());
      d = LIBXSMM_DELTA(t, (double)n);
      if (delta < d) delta = d;
      total += t;
    }
    else {
      total += (double)n;
      ++ninterrupts;
    }
    start = libxsmm_timer_tick();
  }

  start = libxsmm_timer_tick();
  if (EXIT_SUCCESS == timer_sleep(1)) {
    t = libxsmm_timer_duration(start, libxsmm_timer_tick());
    d = LIBXSMM_DELTA(t, 1.0);
    if (delta < d) delta = d;
    total += t;
  }
  else {
    ++ninterrupts;
    total += 1.0;
  }
  start = libxsmm_timer_tick();
  rend = libxsmm_timer_tick_rtc();

  d = LIBXSMM_DELTA(total, (double)max_nseconds);
  if (delta < d) delta = d;

  result = (int)LIBXSMM_ROUND(100.0 * delta);
  if (0 > max_delta || result <= max_delta) {
    const double r = libxsmm_timer_duration_rtc(rbegin, rend);
    libxsmm_cpuid_x86_info info;
    libxsmm_cpuid_x86(&info);
    d = libxsmm_timer_duration(begin, start);
    fprintf(stderr, "seconds=%f..%f delta=%s%i%% interrupted=%u tsc=%sconstant\n",
      r, d, 0 == result ? "" : (total <= d ? "+" : "-"), result, ninterrupts,
      info.constant_tsc ? "" : "non-");
    result = EXIT_SUCCESS;
  }
  else if ((MAX_TOLPERC) >= result) {
    result = EXIT_SUCCESS;
  }

  return result;
}

