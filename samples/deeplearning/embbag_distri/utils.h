/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/

#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_threads() (1)
#define omp_get_thread_num() (0)
#define omp_get_max_threads() (1)
#endif

const int alignment = 64;
typedef long ITyp;
typedef float FTyp;
typedef uint16_t Half;

extern thread_local struct drand48_data rand_buf;

static double get_time() {
  static bool init_done = false;
  static struct timespec stp = {0,0};
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  /*clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tp);*/

  if(!init_done) {
    init_done = true;
    stp = tp;
  }
  double ret = (tp.tv_sec - stp.tv_sec) * 1e3 + (tp.tv_nsec - stp.tv_nsec)*1e-6;
  return ret;
}

void set_random_seed(int seed);

template<typename T>
void init_zero(size_t sz, T *buf)
{
#pragma omp parallel for
  for(size_t i = 0; i < sz; i++)
    buf[i] = (T)0;
}

template<typename T>
void init_random(size_t sz, T *buf, T low, T high)
{
  T range = high - low;
#pragma omp parallel for schedule(static)
  for(size_t i = 0; i < sz; i++) {
    double randval;
    drand48_r(&rand_buf, &randval);
    buf[i] = randval * range - low;
  }
}

inline void *my_malloc(size_t sz, size_t align)
{
    return _mm_malloc(sz, align);
}

inline void my_free(void *p)
{
    _mm_free(p);
}

#endif /*_UTILS_H_*/
