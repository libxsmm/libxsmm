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
#include <libxsmm_timer.h>
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(_WIN32)
# include <Windows.h>
#elif defined(__GNUC__) || defined(__PGI) || defined(_CRAYC)
# include <sys/time.h>
# include <time.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if defined(__powerpc64__)
# include <sys/platform/ppc.h>
#endif

#if !defined(LIBXSMM_TIMER_TSC)
# define LIBXSMM_TIMER_TSC
#endif
#if !defined(LIBXSMM_TIMER_WPC)
# define LIBXSMM_TIMER_WPC
#endif

#if defined(LIBXSMM_TIMER_TSC)
# if defined(__powerpc64__)
#   define LIBXSMM_TIMER_RDTSC(CYCLE) do { \
      CYCLE = __ppc_get_timebase(); \
    } while(0)
# elif ((defined(LIBXSMM_PLATFORM_X86) && (64 <= (LIBXSMM_BITS))) && \
        (defined(__GNUC__) || defined(LIBXSMM_INTEL_COMPILER) || defined(__PGI)))
#   define LIBXSMM_TIMER_RDTSC(CYCLE) do { \
      libxsmm_timer_tickint libxsmm_timer_rdtsc_hi_; \
      __asm__ __volatile__ ("rdtsc" : "=a"(CYCLE), "=d"(libxsmm_timer_rdtsc_hi_)); \
      CYCLE |= libxsmm_timer_rdtsc_hi_ << 32; \
    } while(0)
# elif (defined(_rdtsc) || defined(_WIN32))
#   define LIBXSMM_TIMER_RDTSC(CYCLE) (CYCLE = __rdtsc())
# endif
#endif


LIBXSMM_API_INTERN double libxsmm_timer_duration_rtc(libxsmm_timer_tickint tick0, libxsmm_timer_tickint tick1)
{
  double result = (double)LIBXSMM_DELTA(tick0, tick1);
#if defined(_WIN32)
# if defined(LIBXSMM_TIMER_WPC)
  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);
  result /= (double)frequency.QuadPart;
# else /* low resolution */
  result *= 1E-3;
# endif
#elif defined(CLOCK_MONOTONIC)
  result *= 1E-9;
#else
  result *= 1E-6;
#endif
  return result;
}


LIBXSMM_API_INTERN libxsmm_timer_tickint libxsmm_timer_tick_rtc(void)
{
  libxsmm_timer_tickint result;
#if defined(_WIN32)
# if defined(LIBXSMM_TIMER_WPC)
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  result = (libxsmm_timer_tickint)t.QuadPart;
# else /* low resolution */
  result = (libxsmm_timer_tickint)GetTickCount64();
# endif
#elif defined(CLOCK_MONOTONIC)
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  result = 1000000000ULL * t.tv_sec + t.tv_nsec;
#else
  struct timeval t;
  gettimeofday(&t, 0);
  result = 1000000ULL * t.tv_sec + t.tv_usec;
#endif
  return result;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_GENERIC)
libxsmm_timer_tickint libxsmm_timer_tick_tsc(void)
{
  libxsmm_timer_tickint result;
#if defined(LIBXSMM_TIMER_RDTSC)
  LIBXSMM_TIMER_RDTSC(result);
#else
  result = libxsmm_timer_tick_rtc();
#endif
  return result;
}


LIBXSMM_API int libxsmm_get_timer_info(libxsmm_timer_info* info)
{
  int result;
  if (NULL != info) {
#if defined(LIBXSMM_TIMER_RDTSC)
    if (0 < libxsmm_timer_scale) {
      info->tsc = 1;
    }
# if !defined(LIBXSMM_INIT_COMPLETED)
    else if (2 > libxsmm_ninit) {
      libxsmm_init();
      if (0 < libxsmm_timer_scale) {
        info->tsc = 1;
      }
      else {
        info->tsc = 0;
      }
    }
# endif
    else {
      info->tsc = 0;
    }
#else
    info->tsc = 0;
#endif
    result = EXIT_SUCCESS;
  }
  else {
#if !defined(NDEBUG)
    static int error_once = 0;
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: invalid argument for libxsmm_get_timer_info specified!\n");
    }
#endif
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API libxsmm_timer_tickint libxsmm_timer_tick(void)
{
  libxsmm_timer_tickint result;
#if defined(LIBXSMM_TIMER_RDTSC)
  if (0 < libxsmm_timer_scale) {
    LIBXSMM_TIMER_RDTSC(result);
  }
# if !defined(LIBXSMM_INIT_COMPLETED)
  else if (2 > libxsmm_ninit) {
    libxsmm_init();
    if (0 < libxsmm_timer_scale) {
      LIBXSMM_TIMER_RDTSC(result);
    }
    else {
      result = libxsmm_timer_tick_rtc();
    }
  }
# endif
  else {
    result = libxsmm_timer_tick_rtc();
  }
#else
  result = libxsmm_timer_tick_rtc();
#endif
  return result;
}


LIBXSMM_API double libxsmm_timer_duration(libxsmm_timer_tickint tick0, libxsmm_timer_tickint tick1)
{
  double result;
#if defined(LIBXSMM_TIMER_RDTSC)
  if (0 < libxsmm_timer_scale) {
    result = (double)LIBXSMM_DELTA(tick0, tick1) * libxsmm_timer_scale;
  }
  else
#endif
  {
    result = libxsmm_timer_duration_rtc(tick0, tick1);
  }
  return result;
}


#if defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_timer_ncycles)(libxsmm_timer_tickint* /*ncycles*/, const libxsmm_timer_tickint* /*tick0*/, const libxsmm_timer_tickint* /*tick1*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_timer_ncycles)(libxsmm_timer_tickint* ncycles, const libxsmm_timer_tickint* tick0, const libxsmm_timer_tickint* tick1)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != ncycles && NULL != tick0 && NULL != tick1)
#endif
  {
    *ncycles = libxsmm_timer_ncycles(*tick0, *tick1);
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_timer_ncycles specified!\n");
  }
#endif
}

#endif /*defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/

