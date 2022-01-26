/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/*
 * Copyright (c) 2013-2014, SeisSol Group
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 **/
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <immintrin.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MEMKIND
#include <hbwmalloc.h>
//#define USE_HBM_DOFS
#define USE_HBM_TDOFS
#define USE_HBM_DERS
//#define USE_HBM_CELLLOCAL_LOCAL
//#define USE_HBM_CELLLOCAL_NEIGH
#define USE_HBM_GLOBALDATA
#endif

#ifdef __MIC__
#define __USE_RDTSC
#endif

double derive_cycles_from_time(double time) {
  // first try to read proxy env variable with freq
  char* p_freq;
  double d_freq;
  double cycles = 1.0;
  p_freq = getenv ("SEISSOL_PROXY_FREQUENCY");
  if (p_freq !=NULL ) {
    d_freq = atof(p_freq);
    printf("detected frequency (SEISSOL_PROXY_FREQUENCY): %f\n", d_freq);
    cycles = time * d_freq * 1.0e6;
  } else {
    FILE* fp;
    fp = popen("lscpu | grep MHz | awk '{print $3}'", "r");
    if (fp > 0) {
      char tmp_buffer[20];
      fread(tmp_buffer, 20, 1, fp);
      d_freq = atof(tmp_buffer);
      printf("detected frequency (lscpu): %f\n", d_freq);
      cycles = time * d_freq * 1.0e6;
      pclose(fp);
    } else {
      cycles = 1.0;
      printf("detected frequency (lscpu) FAILED!\n");
    }
  }
  return cycles;
}

// seissol_kernel includes
#include <Initializer/typedefs.hpp>
#include <Kernels/common.hpp>
#include <Time.h>
#include <Volume.h>
#include <Boundary.h>

#include "proxy_seissol_allocator.hpp"
#include "proxy_seissol_flops.hpp"
#include "proxy_seissol_bytes.hpp"
#include "proxy_seissol_integrators.hpp"

inline double sec(struct timeval start, struct timeval end) {
  return ((double)(((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)))) / 1.0e6;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Wrong parameters!\n");
    printf(" #cells #timesteps kernel\n");
    printf("   kernel-values: all, local, neigh, ader, vol, bndlocal\n");
    return -1;
  }

  unsigned int i_cells = atoi(argv[1]);
  unsigned int i_timesteps = atoi(argv[2]);
  std::string s_part;
  s_part.assign(argv[3]);

  // double-check if the selected kernel exists
  if ( (s_part.compare("all") != 0) &&
       (s_part.compare("local") != 0) &&
       (s_part.compare("neigh") != 0) &&
       (s_part.compare("ader") != 0) &&
       (s_part.compare("vol") != 0) &&
       (s_part.compare("bndlocal") != 0) )
  {
    printf("Wrong parameters!\n");
    printf(" #cells #timesteps kernel\n");
    printf("   kernel-values: all, local, neigh, ader, vol, bndlocal\n");
    return -1;
  }

  printf("Allocating fake data...\n");
  i_cells = init_data_structures(i_cells);
  printf("...done\n\n");

  struct timeval start_time, end_time;
  size_t cycles_start, cycles_end;
  double total = 0.0;
  double total_cycles = 0.0;

  // init OpenMP and LLC
  if (s_part.compare("all") == 0) {
    computeLocalIntegration();
    computeNeighboringIntegration();
  } else if (s_part.compare("local") == 0) {
    computeLocalIntegration();
  } else if (s_part.compare("neigh") == 0) {
    computeNeighboringIntegration();
  } else if (s_part.compare("ader") == 0) {
    computeAderIntegration();
  } else if (s_part.compare("vol") == 0) {
    computeVolumeIntegration();
  } else {
    computeLocalBoundaryIntegration();
  }

  gettimeofday(&start_time, NULL);
#ifdef __USE_RDTSC
  cycles_start = _libxsmm_timer_cycles();
#endif

  if (s_part.compare("all") == 0) {
    for (unsigned int t = 0; t < i_timesteps; t++) {
      computeLocalIntegration();
      computeNeighboringIntegration();
    }
  } else if (s_part.compare("local") == 0) {
    for (unsigned int t = 0; t < i_timesteps; t++) {
      computeLocalIntegration();
    }
  } else if (s_part.compare("neigh") == 0) {
    for (unsigned int t = 0; t < i_timesteps; t++) {
      computeNeighboringIntegration();
    }
  } else if (s_part.compare("ader") == 0) {
    for (unsigned int t = 0; t < i_timesteps; t++) {
      computeAderIntegration();
    }
  } else if (s_part.compare("vol") == 0) {
    for (unsigned int t = 0; t < i_timesteps; t++) {
      computeVolumeIntegration();
    }
  } else {
    for (unsigned int t = 0; t < i_timesteps; t++) {
      computeLocalBoundaryIntegration();
    }
  }
#ifdef __USE_RDTSC
  cycles_end = _libxsmm_timer_cycles();
#endif
  gettimeofday(&end_time, NULL);
  total = sec(start_time, end_time);
#ifdef __USE_RDTSC
  printf("Cycles via _libxsmm_timer_cycles()!\n");
  total_cycles = (double)(cycles_end-cycles_start);
#else
  total_cycles = derive_cycles_from_time(total);
#endif

  printf("=================================================\n");
  printf("===            PERFORMANCE SUMMARY            ===\n");
  printf("=================================================\n");
  printf("seissol proxy mode                  : %s\n", s_part.c_str());
  printf("time for seissol proxy              : %f\n", total);
  printf("cycles                              : %f\n\n", total_cycles);
  seissol_flops actual_flops;
  if (s_part.compare("all") == 0) {
    actual_flops = flops_all_actual(i_timesteps);
    printf("GFLOP (non-zero) for seissol proxy  : %f\n", actual_flops.d_nonZeroFlops/(1e9));
    printf("GFLOP (hardware) for seissol proxy  : %f\n", actual_flops.d_hardwareFlops/(1e9));
    //printf("GFLOP (estimate) for seissol proxy  : %f\n", flops_all(i_timesteps)/(1e9));
    printf("GiB (estimate) for seissol proxy    : %f\n\n", bytes_all(i_timesteps)/(1024.0*1024.0*1024.0));
    printf("FLOPS/cycle (non-zero)              : %f\n", actual_flops.d_nonZeroFlops/total_cycles);
    printf("FLOPS/cycle (hardware)              : %f\n", actual_flops.d_hardwareFlops/total_cycles);
    printf("Bytes/cycle (estimate)              : %f\n\n", bytes_all(i_timesteps)/total_cycles);
    printf("GFLOPS (non-zero) for seissol proxy : %f\n", (actual_flops.d_nonZeroFlops/(1e9))/total);
    printf("GFLOPS (hardware) for seissol proxy : %f\n", (actual_flops.d_hardwareFlops/(1e9))/total);
    printf("GiB/s (estimate) for seissol proxy  : %f\n", (bytes_all(i_timesteps)/(1024.0*1024.0*1024.0))/total);
  } else if (s_part.compare("local") == 0) {
    actual_flops = flops_local_actual(i_timesteps);
    printf("GFLOP (non-zero) for seissol proxy  : %f\n", actual_flops.d_nonZeroFlops/(1e9));
    printf("GFLOP (hardware) for seissol proxy  : %f\n", actual_flops.d_hardwareFlops/(1e9));
    //printf("GFLOP (estimate) for seissol proxy  : %f\n", flops_local(i_timesteps)/(1e9));
    printf("GiB (estimate) for seissol proxy    : %f\n\n", bytes_local(i_timesteps)/(1024.0*1024.0*1024.0));
    printf("FLOPS/cycle (non-zero)              : %f\n", actual_flops.d_nonZeroFlops/total_cycles);
    printf("FLOPS/cycle (hardware)              : %f\n", actual_flops.d_hardwareFlops/total_cycles);
    printf("Bytes/cycle (estimate)              : %f\n\n", bytes_local(i_timesteps)/total_cycles);
    printf("GFLOPS (non-zero) for seissol proxy : %f\n", (actual_flops.d_nonZeroFlops/(1e9))/total);
    printf("GFLOPS (hardware) for seissol proxy : %f\n", (actual_flops.d_hardwareFlops/(1e9))/total);
    printf("GiB/s (estimate) for seissol proxy  : %f\n", (bytes_local(i_timesteps)/(1024.0*1024.0*1024.0))/total);
  } else if (s_part.compare("neigh") == 0) {
    actual_flops = flops_bndneigh_actual(i_timesteps);
    printf("GFLOP (non-zero) for seissol proxy  : %f\n", actual_flops.d_nonZeroFlops/(1e9));
    printf("GFLOP (hardware) for seissol proxy  : %f\n", actual_flops.d_hardwareFlops/(1e9));
    //printf("GFLOP (estimate) for seissol proxy  : %f\n", flops_bndneigh(i_timesteps)/(1e9));
    printf("GiB (estimate) for seissol proxy    : %f\n\n", bytes_bndneigh(i_timesteps)/(1024.0*1024.0*1024.0));
    printf("FLOPS/cycle (non-zero)              : %f\n", actual_flops.d_nonZeroFlops/total_cycles);
    printf("FLOPS/cycle (hardware)              : %f\n", actual_flops.d_hardwareFlops/total_cycles);
    printf("Bytes/cycle (estimate)              : %f\n\n", bytes_bndneigh(i_timesteps)/total_cycles);
    printf("GFLOPS (non-zero) for seissol proxy : %f\n", (actual_flops.d_nonZeroFlops/(1e9))/total);
    printf("GFLOPS (hardware) for seissol proxy : %f\n", (actual_flops.d_hardwareFlops/(1e9))/total);
    printf("GiB/s (estimate) for seissol proxy  : %f\n", (bytes_bndneigh(i_timesteps)/(1024.0*1024.0*1024.0))/total);
  } else if (s_part.compare("ader") == 0) {
    actual_flops = flops_ader_actual(i_timesteps);
    printf("GFLOP (non-zero) for seissol proxy  : %f\n", actual_flops.d_nonZeroFlops/(1e9));
    printf("GFLOP (hardware) for seissol proxy  : %f\n", actual_flops.d_hardwareFlops/(1e9));
    //printf("GFLOP (estimate) for seissol proxy  : %f\n", flops_ader(i_timesteps)/(1e9));
    printf("GiB (estimate) for seissol proxy    : %f\n\n", bytes_ader(i_timesteps)/(1024.0*1024.0*1024.0));
    printf("FLOPS/cycle (non-zero)              : %f\n", actual_flops.d_nonZeroFlops/total_cycles);
    printf("FLOPS/cycle (hardware)              : %f\n", actual_flops.d_hardwareFlops/total_cycles);
    printf("Bytes/cycle (estimate)              : %f\n\n", bytes_ader(i_timesteps)/total_cycles);
    printf("GFLOPS (non-zero) for seissol proxy : %f\n", (actual_flops.d_nonZeroFlops/(1e9))/total);
    printf("GFLOPS (hardware) for seissol proxy : %f\n", (actual_flops.d_hardwareFlops/(1e9))/total);
    printf("GiB/s (estimate) for seissol proxy  : %f\n", (bytes_ader(i_timesteps)/(1024.0*1024.0*1024.0))/total);
  } else if (s_part.compare("vol") == 0) {
    actual_flops = flops_vol_actual(i_timesteps);
    printf("GFLOP (non-zero) for seissol proxy  : %f\n", actual_flops.d_nonZeroFlops/(1e9));
    printf("GFLOP (hardware) for seissol proxy  : %f\n", actual_flops.d_hardwareFlops/(1e9));
    //printf("GFLOP (estimate) for seissol proxy  : %f\n", flops_vol(i_timesteps)/(1e9));
    printf("GiB (estimate) for seissol proxy    : %f\n\n", bytes_vol(i_timesteps)/(1024.0*1024.0*1024.0));
    printf("FLOPS/cycle (non-zero)              : %f\n", actual_flops.d_nonZeroFlops/total_cycles);
    printf("FLOPS/cycle (hardware)              : %f\n", actual_flops.d_hardwareFlops/total_cycles);
    printf("Bytes/cycle (estimate)              : %f\n\n", bytes_vol(i_timesteps)/total_cycles);
    printf("GFLOPS (non-zero) for seissol proxy : %f\n", (actual_flops.d_nonZeroFlops/(1e9))/total);
    printf("GFLOPS (hardware) for seissol proxy : %f\n", (actual_flops.d_hardwareFlops/(1e9))/total);
    printf("GiB/s (estimate) for seissol proxy  : %f\n", (bytes_vol(i_timesteps)/(1024.0*1024.0*1024.0))/total);
  } else {
    actual_flops = flops_bndlocal_actual(i_timesteps);
    printf("GFLOP (non-zero) for seissol proxy  : %f\n", actual_flops.d_nonZeroFlops/(1e9));
    printf("GFLOP (hardware) for seissol proxy  : %f\n", actual_flops.d_hardwareFlops/(1e9));
    //printf("GFLOP (estimate) for seissol proxy  : %f\n", flops_bndlocal(i_timesteps)/(1e9));
    printf("GiB (estimate) for seissol proxy    : %f\n\n", bytes_bndlocal(i_timesteps)/(1024.0*1024.0*1024.0));
    printf("FLOPS/cycle (non-zero)              : %f\n", actual_flops.d_nonZeroFlops/total_cycles);
    printf("FLOPS/cycle (hardware)              : %f\n", actual_flops.d_hardwareFlops/total_cycles);
    printf("Bytes/cycle (estimate)              : %f\n\n", bytes_bndlocal(i_timesteps)/total_cycles);
    printf("GFLOPS (non-zero) for seissol proxy : %f\n", (actual_flops.d_nonZeroFlops/(1e9))/total);
    printf("GFLOPS (hardware) for seissol proxy : %f\n", (actual_flops.d_hardwareFlops/(1e9))/total);
    printf("GiB/s (estimate) for seissol proxy  : %f\n", (bytes_bndlocal(i_timesteps)/(1024.0*1024.0*1024.0))/total);
  }
  printf("=================================================\n");
  printf("\n");

  free_data_structures();

  return 0;
}

