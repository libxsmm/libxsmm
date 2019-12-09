/**
 * @file This file is part of EDGE.
 *
 * @author Alexander Breuer (anbreuer AT ucsd.edu)
 *
 * @section LICENSE
 * Copyright (c) 2016, Regents of the University of California
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 * Definition of global variables for parallelization.
 **/
#ifndef GLOBAL_H_
#define GLOBAL_H_

#include "constants.hpp"
#include <string>

namespace edge {
  namespace parallel {
    //! omp-threadprivate thread number
    extern int           g_thread;
    //! omp-threadprivate thread number as string (null-terminated char array)
    extern char          g_threadStr[10];
#ifdef PP_SCRATCH_MEMORY
    //! private scratch high-bandwidth memory for shared memory parallelizations
    extern t_scratchMem* g_scratchMem;
#ifdef PP_USE_OMP
#pragma omp threadprivate(edge::parallel::g_scratchMem)
#endif

#endif

#ifdef PP_USE_OMP
#pragma omp threadprivate(edge::parallel::g_thread, edge::parallel::g_threadStr)
#endif
    //! number of threads
    extern int         g_nThreads;

    //! MPI-rank
    extern int         g_rank;
    //! MPI-rank as string
    extern std::string g_rankStr;
    //! number of MPI-ranks
    extern int         g_nRanks;
  }
}

#ifdef PP_USE_OMP
#pragma omp threadprivate(edge::parallel::g_thread, edge::parallel::g_threadStr)
#endif
int           edge::parallel::g_nThreads = 1;
int           edge::parallel::g_thread = 0;
char          edge::parallel::g_threadStr[10] = "0\000000000";
#ifdef PP_SCRATCH_MEMORY
#ifdef PP_USE_OMP
#pragma omp threadprivate(edge::parallel::g_scratchMem)
#endif
t_scratchMem* edge::parallel::g_scratchMem = nullptr;
#endif

int           edge::parallel::g_nRanks = 1;
int           edge::parallel::g_rank = 0;
std::string   edge::parallel::g_rankStr = std::to_string(0);

#endif
