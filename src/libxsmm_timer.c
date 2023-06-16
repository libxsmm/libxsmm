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
#include <utils/libxsmm_timer.h>
#include "libxsmm_main.h"

#if !defined(LIBXSMM_TIMER_VERBOSE) && !defined(NDEBUG)
# if !defined(LIBXSMM_PLATFORM_AARCH64) || !defined(__APPLE__)
#   define LIBXSMM_TIMER_VERBOSE
# endif
#endif


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
#if defined(LIBXSMM_TIMER_VERBOSE)
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
  if (NULL != ncycles && NULL != tick0 && NULL != tick1)
#endif
  {
    *ncycles = libxsmm_timer_ncycles(*tick0, *tick1);
  }
#if defined(LIBXSMM_TIMER_VERBOSE)
  else if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_timer_ncycles specified!\n");
    }
  }
#endif
}

#endif /*defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/
