/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef PROXY_SEISSOL_INTEGRATORS_HPP
#define PROXY_SEISSOL_INTEGRATORS_HPP

#if defined(_OPENMP)
# include <omp.h>
#endif

void computeAderIntegration() {
#ifdef _OPENMP
# pragma omp parallel
  {
#if NUMBER_OF_THREADS_PER_GLOBALDATA_COPY < 512
  //GlobalData* l_globalData = m_globalDataArray[omp_get_thread_num()/NUMBER_OF_THREADS_PER_GLOBALDATA_COPY];
  GlobalData* l_globalData = m_globalDataArray[0];
#else
  GlobalData* l_globalData = m_globalData;
#endif
  #pragma omp for schedule(static)
#else
  GlobalData* l_globalData = m_globalData;
#endif
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    m_timeKernel.computeAder(              m_timeStepWidthSimulation,
                                           l_globalData->stiffnessMatricesTransposed,
                                           m_cells->dofs[l_cell],
                                           m_cellData->localIntegration[l_cell].starMatrices,
                                           m_cells->buffers[l_cell],
                                           m_cells->derivatives[l_cell] );
  }
#ifdef _OPENMP
  }
#endif
}

void computeVolumeIntegration() {
#ifdef _OPENMP
# pragma omp parallel
  {
#if NUMBER_OF_THREADS_PER_GLOBALDATA_COPY < 512
  //GlobalData* l_globalData = m_globalDataArray[omp_get_thread_num()/NUMBER_OF_THREADS_PER_GLOBALDATA_COPY];
  GlobalData* l_globalData = m_globalDataArray[0];
#else
  GlobalData* l_globalData = m_globalData;
#endif
  #pragma omp for schedule(static)
#else
  GlobalData* l_globalData = m_globalData;
#endif
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    m_volumeKernel.computeIntegral(        l_globalData->stiffnessMatrices,
                                           m_cells->buffers[l_cell],
                                           m_cellData->localIntegration[l_cell].starMatrices,
                                           m_cells->dofs[l_cell] );
  }
#ifdef _OPENMP
  }
#endif
}

void computeLocalBoundaryIntegration() {
#ifdef _OPENMP
  #pragma omp parallel
  {
#if NUMBER_OF_THREADS_PER_GLOBALDATA_COPY < 512
  //GlobalData* l_globalData = m_globalDataArray[omp_get_thread_num()/NUMBER_OF_THREADS_PER_GLOBALDATA_COPY];
  GlobalData* l_globalData = m_globalDataArray[0];
#else
  GlobalData* l_globalData = m_globalData;
#endif
  #pragma omp for schedule(static)
#else
  GlobalData* l_globalData = m_globalData;
#endif
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    m_boundaryKernel.computeLocalIntegral( m_cellInformation[l_cell].faceTypes,
                                           l_globalData->fluxMatrices,
                                           m_cells->buffers[l_cell],
                                           m_cellData->localIntegration[l_cell].nApNm1,
#ifdef ENABLE_STREAM_MATRIX_PREFETCH
                                           m_cells->dofs[l_cell],
                                           m_cells->buffers[l_cell+1],
                                           m_cells->dofs[l_cell+1] );
#else
                                           m_cells->dofs[l_cell] );
#endif
  }
#ifdef _OPENMP
  }
#endif
}

void computeLocalIntegration() {
#ifdef _OPENMP
  #pragma omp parallel
  {
#if NUMBER_OF_THREADS_PER_GLOBALDATA_COPY < 512
  //GlobalData* l_globalData = m_globalDataArray[omp_get_thread_num()/NUMBER_OF_THREADS_PER_GLOBALDATA_COPY];
  GlobalData* l_globalData = m_globalDataArray[0];
#else
  GlobalData* l_globalData = m_globalData;
#endif
  #pragma omp for schedule(static)
#else
  GlobalData* l_globalData = m_globalData;
#endif
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    m_timeKernel.computeAder(      (double)m_timeStepWidthSimulation,
                                           l_globalData->stiffnessMatricesTransposed,
                                           m_cells->dofs[l_cell],
                                           m_cellData->localIntegration[l_cell].starMatrices,
                                           m_cells->buffers[l_cell],
                                           m_cells->derivatives[l_cell] );

    m_volumeKernel.computeIntegral(        l_globalData->stiffnessMatrices,
                                           m_cells->buffers[l_cell],
                                           m_cellData->localIntegration[l_cell].starMatrices,
                                           m_cells->dofs[l_cell] );

    m_boundaryKernel.computeLocalIntegral( m_cellInformation[l_cell].faceTypes,
                                           l_globalData->fluxMatrices,
                                           m_cells->buffers[l_cell],
                                           m_cellData->localIntegration[l_cell].nApNm1,
#ifdef ENABLE_STREAM_MATRIX_PREFETCH
                                           m_cells->dofs[l_cell],
                                           m_cells->buffers[l_cell+1],
                                           m_cells->dofs[l_cell+1] );
#else
                                           m_cells->dofs[l_cell] );
#endif
  }
#ifdef _OPENMP
  }
#endif
}

void computeNeighboringIntegration() {
  real  l_integrationBuffer[4][NUMBER_OF_ALIGNED_DOFS] __attribute__((aligned(4096)));
  real *l_timeIntegrated[4];
#ifdef ENABLE_MATRIX_PREFETCH
  real *l_faceNeighbors_prefetch[4];
  real *l_fluxMatricies_prefetch[4];
#endif

#ifdef _OPENMP
#ifdef ENABLE_MATRIX_PREFETCH
  #pragma omp parallel private(l_integrationBuffer, l_timeIntegrated, l_faceNeighbors_prefetch, l_fluxMatricies_prefetch)
#else
  #pragma omp parallel private(l_integrationBuffer, l_timeIntegrated)
#endif
  {
#if NUMBER_OF_THREADS_PER_GLOBALDATA_COPY < 512
  GlobalData* l_globalData = m_globalDataArray[omp_get_thread_num()/NUMBER_OF_THREADS_PER_GLOBALDATA_COPY];
#else
  GlobalData* l_globalData = m_globalData;
#endif
  #pragma omp for schedule(static)
#else
  GlobalData* l_globalData = m_globalData;
#endif
  for( int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    m_timeKernel.computeIntegrals(             m_cellInformation[l_cell].ltsSetup,
                                               m_cellInformation[l_cell].faceTypes,
                                               m_cellInformation[l_cell].currentTime,
                                       (double)m_timeStepWidthSimulation,
                                               m_cells->faceNeighbors[l_cell],
                                               l_integrationBuffer,
                                               l_timeIntegrated );

#ifdef ENABLE_MATRIX_PREFETCH
#pragma message("the current prefetch structure (flux matrices and tDOFs is tuned for higher order and shouldn't be harmful for lower orders")
    int l_face = 1;
    l_faceNeighbors_prefetch[0] = m_cells->faceNeighbors[l_cell][l_face];
    l_fluxMatricies_prefetch[0] = l_globalData->fluxMatrices[4+(l_face*12)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][0]*3)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][1])];
    l_face = 2;
    l_faceNeighbors_prefetch[1] = m_cells->faceNeighbors[l_cell][l_face];
    l_fluxMatricies_prefetch[1] = l_globalData->fluxMatrices[4+(l_face*12)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][0]*3)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][1])];
    l_face = 3;
    l_faceNeighbors_prefetch[2] = m_cells->faceNeighbors[l_cell][l_face];
    l_fluxMatricies_prefetch[2] = l_globalData->fluxMatrices[4+(l_face*12)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][0]*3)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][1])];
    l_face = 0;
    if (l_cell < (m_cells->numberOfCells-1) ) {
      l_faceNeighbors_prefetch[3] = m_cells->faceNeighbors[l_cell+1][l_face];
      l_fluxMatricies_prefetch[3] = l_globalData->fluxMatrices[4+(l_face*12)
                                                               +(m_cellInformation[l_cell+1].faceRelations[l_face][0]*3)
                                                               +(m_cellInformation[l_cell+1].faceRelations[l_face][1])];
    } else {
      l_faceNeighbors_prefetch[3] = m_cells->faceNeighbors[l_cell][3];
      l_fluxMatricies_prefetch[3] = l_globalData->fluxMatrices[4+(3*12)
                                                               +(m_cellInformation[l_cell].faceRelations[l_face][0]*3)
                                                               +(m_cellInformation[l_cell].faceRelations[l_face][1])];
    }
#endif

    m_boundaryKernel.computeNeighborsIntegral( m_cellInformation[l_cell].faceTypes,
                                               m_cellInformation[l_cell].faceRelations,
                                               l_globalData->fluxMatrices,
                                               l_timeIntegrated,
                                               m_cellData->neighboringIntegration[l_cell].nAmNm1,
#ifdef ENABLE_MATRIX_PREFETCH
                                               m_cells->dofs[l_cell],
                                               l_faceNeighbors_prefetch,
                                               l_fluxMatricies_prefetch );
#else
                                               m_cells->dofs[l_cell]);
#endif
  }

#ifdef _OPENMP
  }
#endif
}

#endif /*PROXY_SEISSOL_INTEGRATORS_HPP*/

