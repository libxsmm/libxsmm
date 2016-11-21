/******************************************************************************
** Copyright (c) 2009-2016, Intel Corporation                                **
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
#include <libxsmm_timer.h>
#include <libxsmm_intrinsics_x86.h>

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

#if (defined(__GNUC__) || defined(__INTEL_COMPILER)) && \
  ((4294967295U < (__SIZE_MAX__)) || defined(_WIN64) || defined(_CRAYC))
# define LIBXSMM_TIMER_RDTSC(CYCLE) { unsigned long long libxsmm_timer_rdtsc_hi_; \
    __asm__ __volatile__ ("rdtsc" : "=a"(CYCLE), "=d"(libxsmm_timer_rdtsc_hi_)); \
    CYCLE |= libxsmm_timer_rdtsc_hi_ << 32; \
  }
#elif defined(_rdtsc) || defined(_WIN32)
# define LIBXSMM_TIMER_RDTSC(CYCLE) (CYCLE = __rdtsc())
#endif


LIBXSMM_API_DEFINITION unsigned long long libxsmm_timer_tick(void)
{
#if defined(_WIN32)
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return (unsigned long long)t.QuadPart;
#elif defined(CLOCK_MONOTONIC)
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return 1000000000ULL * t.tv_sec + t.tv_nsec;
#else
  struct timeval t;
  gettimeofday(&t, 0);
  return 1000000ULL * t.tv_sec + t.tv_usec;
#endif
}


LIBXSMM_API_DEFINITION /*LIBXSMM_INTRINSICS*/ unsigned long long libxsmm_timer_xtick(void)
{
  unsigned long long result;
#if defined(LIBXSMM_TIMER_RDTSC)
  LIBXSMM_TIMER_RDTSC(result);
#else
  LIBXSMM_MESSAGE("================================================================================")
  LIBXSMM_MESSAGE("LIBXSMM: Support for the RDTSC intrinsic appears to be unavailable!")
  LIBXSMM_MESSAGE("================================================================================")
  result = libxsmm_timer_tick();
#endif
  return result;
}


LIBXSMM_API_DEFINITION double libxsmm_timer_duration(unsigned long long tick0, unsigned long long tick1)
{
  const double d = (double)(LIBXSMM_MAX(tick1, tick0) - tick0);
#if defined(_WIN32)
  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);
  return d / (double)frequency.QuadPart;
#elif defined(CLOCK_MONOTONIC)
  return d * 1E-9;
#else
  return d * 1E-6;
#endif
}

