/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Ishwar Bhati (Intel Corp.)
   Dhiraj Kalamkar (Intel Corp.)
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


static double get_time() {
  static bool init_done = false;
  static struct timespec stp = {0,0};
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  /*clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &tp);*/

  if (!init_done) {
    init_done = true;
    stp = tp;
  }
  double ret = (tp.tv_sec - stp.tv_sec) * 1e3 + (tp.tv_nsec - stp.tv_nsec)*1e-6;
  return ret;
}

#endif /*_UTILS_H_*/
