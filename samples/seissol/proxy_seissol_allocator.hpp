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
#ifndef PROXY_SEISSOL_ALLOCATOR_HPP
#define PROXY_SEISSOL_ALLOCATOR_HPP

struct CellLocalInformation {
  enum faceType faceTypes[4];
  int faceRelations[4][2];
  unsigned int faceNeighborIds[4];
  unsigned int ltsSetup;
  double currentTime[5];
};

struct GlobalData {
  real *stiffnessMatricesTransposed[3];
  real *stiffnessMatrices[3];
  real *fluxMatrices[52];
};

struct LocalIntegrationData {
  real starMatrices[3][STAR_NNZ];
  real nApNm1[4][NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
};

struct NeighboringIntegrationData {
  real nAmNm1[4][NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES];
};

struct CellData {
  struct LocalIntegrationData       *localIntegration;
  struct NeighboringIntegrationData *neighboringIntegration;
};

struct Cells {
  unsigned int numberOfCells;
  real (*dofs)[NUMBER_OF_ALIGNED_DOFS];
  real **buffers;
  real **derivatives;
  real *(*faceNeighbors)[4];
};

struct GlobalData **m_globalDataArray;
struct GlobalData *m_globalData;
struct CellLocalInformation *m_cellInformation;
struct CellData *m_cellData;
struct Cells *m_cells;
struct LocalIntegrationData *m_localIntegration;
struct NeighboringIntegrationData * m_neighboringIntegration;

seissol::kernels::Time     m_timeKernel;
seissol::kernels::Volume   m_volumeKernel;
seissol::kernels::Boundary m_boundaryKernel;

/* This option is needed to avoid pollution of low-level caches */
#define NUMBER_OF_THREADS_PER_GLOBALDATA_COPY 4
#ifndef NUMBER_OF_THREADS_PER_GLOBALDATA_COPY
#define NUMBER_OF_THREADS_PER_GLOBALDATA_COPY 16383
#endif

real m_timeStepWidthSimulation = (real)1.0;
real* m_dofs;
real* m_tdofs;
#ifdef __USE_DERS
real* m_ders;
#endif
real** m_ptdofs;
real** m_pder;
real* m_faceNeighbors;
real** m_globalPointerArray;
real* m_globalPointer;

unsigned int init_data_structures(unsigned int i_cells) {
  // check if we have to read on scenario
  char* pScenario;
  std::string s_scenario;
  bool bUseScenario = false;
  unsigned int (*scenario_faceType)[4];
  unsigned int (*scenario_neighbor)[4];
  unsigned int (*scenario_side)[4];
  unsigned int (*scenario_orientation)[4];

  pScenario = getenv ("SEISSOL_PROXY_SCENARIO");
  if (pScenario !=NULL ) {
    bUseScenario = true;
    s_scenario.assign(pScenario);
    std::string file;
    std::ifstream data;
    size_t reads;
    unsigned int value;

    // read scenario size
    file = s_scenario + ".size";
    data.open(file.c_str());
    reads = 0;
    if (!data) { printf("size of scenario couldn't be read!\n"); exit(-1); }
    while (data >> i_cells) {
      printf("scenario name is: %s\n", s_scenario.c_str());
      printf("scenario has %i cells\n", i_cells);
      reads++;
    }
    data.close();
    if (reads != 1) { printf("wrong number of sizes (%i) in scenario were read!\n", reads); exit(-1); }

    scenario_neighbor = (unsigned int(*)[4]) malloc(i_cells*sizeof(unsigned int[4]));
    scenario_faceType = (unsigned int(*)[4]) malloc(i_cells*sizeof(unsigned int[4]));
    scenario_side = (unsigned int(*)[4]) malloc(i_cells*sizeof(unsigned int[4]));
    scenario_orientation = (unsigned int(*)[4]) malloc(i_cells*sizeof(unsigned int[4]));

    // read neighbors
    file = s_scenario + ".neigh";
    data.open(file.c_str());
    if (!data) { printf("neigh of scenario couldn't be read!\n"); exit(-1); }
    reads = 0;
    while (data >> value) {
      scenario_neighbor[reads/4][reads%4] = value;
      reads++;
    }
    data.close();
    if (reads != i_cells*4) { printf("wrong neigh (%i) in scenario were read!\n", reads); exit(-1); }

    // read faceTypes
    file = s_scenario + ".bound";
    data.open(file.c_str());
    if (!data) { printf("bound of scenario couldn't be read!\n"); exit(-1); }
    reads = 0;
    while (data >> value) {
      scenario_faceType[reads/4][reads%4] = value;
      reads++;
    }
    data.close();
    if (reads != i_cells*4) { printf("wrong faceType (%i) in scenario were read!\n", reads); exit(-1); }

    // read sides
    file = s_scenario + ".sides";
    data.open(file.c_str());
    if (!data) { printf("sides of scenario couldn't be read!\n"); exit(-1); }
    reads = 0;
    while (data >> value) {
      scenario_side[reads/4][reads%4] = value;
      reads++;
    }
    data.close();
    if (reads != i_cells*4) { printf("wrong sides (%i) in scenario were read!\n", reads); exit(-1); }

    // read orientation
    file = s_scenario + ".orient";
    data.open(file.c_str());
    if (!data) { printf("orientations of scenario couldn't be read!\n"); exit(-1); }
    reads = 0;
    while (data >> value) {
      scenario_orientation[reads/4][reads%4] = value;
      reads++;
    }
    data.close();
    if (reads != i_cells*4) { printf("wrong orientations (%i) in scenario were read!\n", reads); exit(-1); }
  }

  // init RNG
  libxsmm_rng_set_seed(i_cells);

  // cell information
  m_cellInformation = (CellLocalInformation*)malloc(i_cells*sizeof(CellLocalInformation));
  for (unsigned int l_cell = 0; l_cell < i_cells; l_cell++) {
    for (unsigned int f = 0; f < 4; f++) {
      if (bUseScenario == true ) {
        switch (scenario_faceType[l_cell][f]) {
          case 0:
            m_cellInformation[l_cell].faceTypes[f] = regular;
            break;
          case 1:
            m_cellInformation[l_cell].faceTypes[f] = freeSurface;
            break;
          case 3:
            m_cellInformation[l_cell].faceTypes[f] = dynamicRupture;
            break;
          case 5:
            m_cellInformation[l_cell].faceTypes[f] = outflow;
            break;
          case 6:
            m_cellInformation[l_cell].faceTypes[f] = periodic;
            break;
          default:
            printf("unsupported faceType (%i)!\n", scenario_faceType[l_cell][f]);
            exit(-1);
            break;
        }
        m_cellInformation[l_cell].faceRelations[f][0] = scenario_side[l_cell][f];
        m_cellInformation[l_cell].faceRelations[f][1] = scenario_orientation[l_cell][f];
        m_cellInformation[l_cell].faceNeighborIds[f] =  scenario_neighbor[l_cell][f];
      } else {
        m_cellInformation[l_cell].faceTypes[f] = regular;
        m_cellInformation[l_cell].faceRelations[f][0] = (libxsmm_rng_u32(4));
        m_cellInformation[l_cell].faceRelations[f][1] = (libxsmm_rng_u32(3));
        m_cellInformation[l_cell].faceNeighborIds[f] = (libxsmm_rng_u32(i_cells));
      }
    }
#ifdef __USE_DERS
    m_cellInformation[l_cell].ltsSetup = 4095;
#else
    m_cellInformation[l_cell].ltsSetup = 0;
#endif
    for (unsigned int f = 0; f < 5; f++) {
      m_cellInformation[l_cell].currentTime[f] = 0.0;
    }
  }

  // DOFs, tIntegrated buffer
#ifdef USE_HBM_DOFS
  hbw_posix_memalign( (void**) &m_dofs, 2097152, sizeof(real[NUMBER_OF_ALIGNED_DOFS])*i_cells );
#else
  posix_memalign( (void**) &m_dofs, 2097152, sizeof(real[NUMBER_OF_ALIGNED_DOFS])*i_cells );
#endif

#ifdef USE_HBM_TDOFS
  hbw_posix_memalign( (void**) &m_tdofs, 2097152, sizeof(real[NUMBER_OF_ALIGNED_DOFS])*i_cells );
#else
  posix_memalign( (void**) &m_tdofs, 2097152, sizeof(real[NUMBER_OF_ALIGNED_DOFS])*i_cells );
#endif

#ifdef __USE_DERS
#ifdef USE_HBM_DERS
  hbw_posix_memalign( (void**) &m_ders, 2097152, sizeof(real[NUMBER_OF_ALIGNED_DERS])*i_cells );
#else
  posix_memalign( (void**) &m_ders, 2097152, sizeof(real[NUMBER_OF_ALIGNED_DERS])*i_cells );
#endif
#endif

  m_ptdofs = (real**)malloc(sizeof(real*)*i_cells);
  m_pder = (real**)malloc(sizeof(real*)*i_cells);
  m_faceNeighbors = (real*)malloc(sizeof(real*[4])*i_cells);

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (unsigned int l_cell = 0; l_cell < i_cells; l_cell++) {
    for (unsigned int i = 0; i < NUMBER_OF_ALIGNED_DOFS; i++) {
      m_dofs[(l_cell*NUMBER_OF_ALIGNED_DOFS)+i] = (real)libxsmm_rng_f64();
    }
    for (unsigned int i = 0; i < NUMBER_OF_ALIGNED_DOFS; i++) {
      m_tdofs[(l_cell*NUMBER_OF_ALIGNED_DOFS)+i] = (real)libxsmm_rng_f64();
    }
  }
#ifdef __USE_DERS
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (unsigned int l_cell = 0; l_cell < i_cells; l_cell++) {
    for (unsigned int i = 0; i < NUMBER_OF_ALIGNED_DERS; i++) {
      m_ders[(l_cell*NUMBER_OF_ALIGNED_DERS)+i] = (real)libxsmm_rng_f64();
    }
  }
#endif

  for (unsigned int l_cell = 0; l_cell < i_cells; l_cell++) {
    m_ptdofs[l_cell] = &(m_tdofs[(l_cell*NUMBER_OF_ALIGNED_DOFS)]);
#ifdef __USE_DERS
    m_pder[l_cell] = &(m_ders[(l_cell*NUMBER_OF_ALIGNED_DERS)]);
#else
    m_pder[l_cell] = NULL;
#endif
  }

  m_cells = (Cells*)malloc(sizeof(Cells));
  m_cells->numberOfCells = i_cells;
  m_cells->dofs = (real(*)[NUMBER_OF_ALIGNED_DOFS])m_dofs;
  m_cells->buffers = m_ptdofs;
  m_cells->derivatives = m_pder;
  m_cells->faceNeighbors = (real*(*)[4])m_faceNeighbors;

  for (unsigned int l_cell = 0; l_cell < i_cells; l_cell++) {
    for (unsigned int f = 0; f < 4; f++) {
      if (m_cellInformation[l_cell].faceTypes[f] == outflow) {
        m_cells->faceNeighbors[l_cell][f] = NULL;
      } else if (m_cellInformation[l_cell].faceTypes[f] == freeSurface) {
#ifdef __USE_DERS
        m_cells->faceNeighbors[l_cell][f] = m_cells->derivatives[l_cell];
#else
        m_cells->faceNeighbors[l_cell][f] = m_cells->buffers[l_cell];
#endif
      } else if (m_cellInformation[l_cell].faceTypes[f] == periodic || m_cellInformation[l_cell].faceTypes[f] == regular) {
#ifdef __USE_DERS
        m_cells->faceNeighbors[l_cell][f] = m_cells->derivatives[m_cellInformation[l_cell].faceNeighborIds[f]];
#else
        m_cells->faceNeighbors[l_cell][f] = m_cells->buffers[m_cellInformation[l_cell].faceNeighborIds[f]];
#endif
      } else {
        printf("unsupported boundary type -> exit\n");
        exit(-1);
      }
    }
  }

  // local integration
#ifdef USE_HBM_CELLLOCAL_LOCAL
  hbw_posix_memalign( (void**) &m_localIntegration, 2097152, i_cells*sizeof(LocalIntegrationData) );
#else
  posix_memalign( (void**) &m_localIntegration, 2097152, i_cells*sizeof(LocalIntegrationData) );
#endif

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (unsigned int l_cell = 0; l_cell < i_cells; l_cell++) {
    // init star matrices
    for (size_t m = 0; m < 3; m++) {
      for (size_t j = 0; j < STAR_NNZ; j++) {
        m_localIntegration[l_cell].starMatrices[m][j] = (real)libxsmm_rng_f64();
      }
    }
    // init flux solver
    for (size_t m = 0; m < 4; m++) {
      for (size_t j = 0; j < NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES; j++) {
        m_localIntegration[l_cell].nApNm1[m][j] = (real)libxsmm_rng_f64();
      }
    }
  }

  // neighbor integration
#ifdef USE_HBM_CELLLOCAL_NEIGH
  hbw_posix_memalign( (void**) &m_neighboringIntegration, 2097152, i_cells*sizeof(NeighboringIntegrationData) );
#else
  posix_memalign( (void**) &m_neighboringIntegration, 2097152, i_cells*sizeof(NeighboringIntegrationData) );
#endif

#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for (unsigned int l_cell = 0; l_cell < i_cells; l_cell++) {
    // init flux solver
    for (size_t m = 0; m < 4; m++) {
      for (size_t j = 0; j < NUMBER_OF_QUANTITIES*NUMBER_OF_QUANTITIES; j++) {
        m_neighboringIntegration[l_cell].nAmNm1[m][j] = (real)libxsmm_rng_f64();
      }
    }
  }

  // CellData
  m_cellData = (CellData*)malloc(sizeof(CellData));
  m_cellData->localIntegration = m_localIntegration;
  m_cellData->neighboringIntegration = m_neighboringIntegration;

  // Global matrices
  unsigned int l_globalMatrices  = NUMBER_OF_ALIGNED_BASIS_FUNCTIONS * seissol::kernels::getNumberOfBasisFunctions( CONVERGENCE_ORDER-1 ) * 3;
               l_globalMatrices += seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER-1 ) * NUMBER_OF_BASIS_FUNCTIONS  * 3;
               l_globalMatrices += NUMBER_OF_ALIGNED_BASIS_FUNCTIONS * NUMBER_OF_BASIS_FUNCTIONS * 52;
               l_globalMatrices *= sizeof(real);

  // determine number of global data copies
  unsigned int l_numberOfThreads = 1;
#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp master
    {
      l_numberOfThreads = omp_get_num_threads();
    }
  }
#endif
  unsigned int l_numberOfCopiesCeil = (l_numberOfThreads%NUMBER_OF_THREADS_PER_GLOBALDATA_COPY == 0) ? 0 : 1;
  unsigned int l_numberOfCopies = (l_numberOfThreads/NUMBER_OF_THREADS_PER_GLOBALDATA_COPY) + l_numberOfCopiesCeil;

  m_globalPointerArray = (real**) malloc(l_numberOfCopies*sizeof(real*));
  m_globalDataArray = (GlobalData**) malloc(l_numberOfCopies*sizeof(GlobalData*));

  // TODO: for NUMA we need to bind this
  for (unsigned int l_globalDataCount = 0; l_globalDataCount < l_numberOfCopies; l_globalDataCount++) {
#ifdef USE_HBM_GLOBALDATA
    hbw_posix_memalign( (void**) &(m_globalPointerArray[l_globalDataCount]), 2097152, l_globalMatrices );
#else
    posix_memalign( (void**) &(m_globalPointerArray[l_globalDataCount]), 2097152, l_globalMatrices );
#endif
    m_globalPointer = m_globalPointerArray[l_globalDataCount];
    m_globalDataArray[l_globalDataCount] = (GlobalData*) malloc(sizeof(GlobalData));
    m_globalData =  m_globalDataArray[l_globalDataCount];

    for (unsigned int i = 0; i < (l_globalMatrices/sizeof(real)); i++) {
      m_globalPointer[i] = (real)libxsmm_rng_f64();
    }

    real* tmp_pointer = m_globalPointer;
    // stiffness for time integration
    for ( unsigned int l_transposedStiffnessMatrix = 0; l_transposedStiffnessMatrix < 3; l_transposedStiffnessMatrix++ ) {
      m_globalData->stiffnessMatricesTransposed[l_transposedStiffnessMatrix] = tmp_pointer;
      tmp_pointer += seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER-1 ) * NUMBER_OF_BASIS_FUNCTIONS;
    }

    // stiffness for volume integration
    for ( unsigned int l_stiffnessMatrix = 0; l_stiffnessMatrix < 3; l_stiffnessMatrix++ ) {
      m_globalData->stiffnessMatrices[l_stiffnessMatrix] = tmp_pointer;
      tmp_pointer += NUMBER_OF_ALIGNED_BASIS_FUNCTIONS * seissol::kernels::getNumberOfBasisFunctions( CONVERGENCE_ORDER-1 );
    }

    // flux matrices for boundary integration
    for ( unsigned int l_fluxMatrix = 0; l_fluxMatrix < 52; l_fluxMatrix++ ) {
      m_globalData->fluxMatrices[l_fluxMatrix] = tmp_pointer;
      tmp_pointer += NUMBER_OF_ALIGNED_BASIS_FUNCTIONS * NUMBER_OF_BASIS_FUNCTIONS;
    }
  }

  // set default to first chunk
  m_globalPointer = m_globalPointerArray[0];
  m_globalData = m_globalDataArray[0];

  if (bUseScenario == true ) {
    free(scenario_faceType);
    free(scenario_neighbor);
    free(scenario_side);
    free(scenario_orientation);
  }

  return i_cells;
}

void free_data_structures() {
  unsigned int l_numberOfThreads = 1;
#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp master
    {
      l_numberOfThreads = omp_get_num_threads();
    }
  }
#endif
  unsigned int l_numberOfCopiesCeil = (l_numberOfThreads%NUMBER_OF_THREADS_PER_GLOBALDATA_COPY == 0) ? 0 : 1;
  unsigned int l_numberOfCopies = (l_numberOfThreads/NUMBER_OF_THREADS_PER_GLOBALDATA_COPY) + l_numberOfCopiesCeil;

  for (unsigned int l_globalDataCount = 0; l_globalDataCount < l_numberOfCopies; l_globalDataCount++) {
    m_globalData =  m_globalDataArray[l_globalDataCount];
    free(m_globalData);
  }
  free(m_globalDataArray);
  free(m_cellInformation);
  free(m_cellData);
  free(m_cells);

#ifdef USE_HBM_CELLLOCAL_LOCAL
  hbw_free(m_localIntegration);
#else
  free(m_localIntegration);
#endif

#ifdef USE_HBM_CELLLOCAL_NEIGH
  hbw_free(m_neighboringIntegration);
#else
  free(m_neighboringIntegration);
#endif

#ifdef USE_HBM_DOFS
  hbw_free(m_dofs);
#else
  free(m_dofs);
#endif

#ifdef USE_HBM_TDOFS
  hbw_free(m_tdofs);
#else
  free(m_tdofs);
#endif

#ifdef __USE_DERS
#ifdef USE_HBM_DERS
  hbw_free(m_ders);
#else
  free(m_ders);
#endif
#endif

  free(m_ptdofs);
  free(m_pder);
  free(m_faceNeighbors);

  for (unsigned int l_globalDataCount = 0; l_globalDataCount < l_numberOfCopies; l_globalDataCount++) {
    m_globalPointer = m_globalPointerArray[l_globalDataCount];
#ifdef USE_HBM_GLOBALDATA
    hbw_free(m_globalPointer);
#else
    free(m_globalPointer);
#endif
  }

  free(m_globalPointerArray);
}

#endif /*PROXY_SEISSOL_ALLOCATOR_HPP*/

