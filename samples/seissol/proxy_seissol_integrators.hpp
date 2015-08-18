/*
Copyright (c) 2015, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <omp.h>

void computeAderIntegration() {
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    m_timeKernel.computeAder(              m_timeStepWidthSimulation,
                                           m_globalData->stiffnessMatricesTransposed,
                                           m_cells->dofs[l_cell],
                                           m_cellData->localIntegration[l_cell].starMatrices,
                                           m_cells->buffers[l_cell],
                                           m_cells->derivatives[l_cell] );
  }
}

void computeVolumeIntegration() {
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    m_volumeKernel.computeIntegral(        m_globalData->stiffnessMatrices,
                                           m_cells->buffers[l_cell],
                                           m_cellData->localIntegration[l_cell].starMatrices,
                                           m_cells->dofs[l_cell] );
  }
}

void computeLocalBoundaryIntegration() {
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    m_boundaryKernel.computeLocalIntegral( m_cellInformation[l_cell].faceTypes,
                                           m_globalData->fluxMatrices,
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
}

void computeLocalIntegration() {
#ifdef _OPENMP
  #pragma omp parallel for schedule(static)
#endif
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    m_timeKernel.computeAder(      (double)m_timeStepWidthSimulation,
                                           m_globalData->stiffnessMatricesTransposed,
                                           m_cells->dofs[l_cell],
                                           m_cellData->localIntegration[l_cell].starMatrices,
                                           m_cells->buffers[l_cell],
                                           m_cells->derivatives[l_cell] );

    m_volumeKernel.computeIntegral(        m_globalData->stiffnessMatrices,
                                           m_cells->buffers[l_cell],
                                           m_cellData->localIntegration[l_cell].starMatrices,
                                           m_cells->dofs[l_cell] );

    m_boundaryKernel.computeLocalIntegral( m_cellInformation[l_cell].faceTypes,
                                           m_globalData->fluxMatrices,
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
}

void computeNeighboringIntegration() {
  real  l_integrationBuffer[4][NUMBER_OF_ALIGNED_DOFS] __attribute__((aligned(4096)));
  real *l_timeIntegrated[4];
  real *l_faceNeighbors_prefetch[4];
  real *l_fluxMatricies_prefetch[4];

#ifdef _OPENMP
  #pragma omp parallel private(l_integrationBuffer, l_timeIntegrated, l_faceNeighbors_prefetch, l_fluxMatricies_prefetch)
  {
  GlobalData* l_globalData = m_globalDataArray[omp_get_thread_num()%NUMBER_OF_GLOBAL_DATA_COPIES];
  //GlobalData* l_globalData = m_globalData;

  #pragma omp for schedule(static)
#endif
  for( int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    m_timeKernel.computeIntegrals(             m_cellInformation[l_cell].ltsSetup,
                                               m_cellInformation[l_cell].faceTypes,
                                               m_cellInformation[l_cell].currentTime,
                                       (double)m_timeStepWidthSimulation,
                                               m_cells->faceNeighbors[l_cell],
                                               l_integrationBuffer,
                                               l_timeIntegrated );
    
#if CONVERGENCE_ORDER == 2
    l_faceNeighbors_prefetch[0] = m_cells->faceNeighbors[l_cell+8][0];
    l_faceNeighbors_prefetch[1] = m_cells->faceNeighbors[l_cell+8][1];
    l_faceNeighbors_prefetch[2] = m_cells->faceNeighbors[l_cell+8][2];
    l_faceNeighbors_prefetch[3] = m_cells->faceNeighbors[l_cell+8][3];

    l_fluxMatricies_prefetch[0] = NULL;
    l_fluxMatricies_prefetch[1] = NULL;
    l_fluxMatricies_prefetch[2] = NULL;
    l_fluxMatricies_prefetch[3] = NULL;
#elif CONVERGENCE_ORDER == 6
#ifdef __USE_DERS
    if (l_cell < (m_cells->numberOfCells-1) ) {
      l_faceNeighbors_prefetch[0] = m_cells->faceNeighbors[l_cell+1][0];
      l_faceNeighbors_prefetch[1] = m_cells->faceNeighbors[l_cell+1][1];
      l_faceNeighbors_prefetch[2] = m_cells->faceNeighbors[l_cell+1][2];
      l_faceNeighbors_prefetch[3] = m_cells->faceNeighbors[l_cell+1][3];
    } else {
      l_faceNeighbors_prefetch[0] = m_cells->faceNeighbors[l_cell][0];
      l_faceNeighbors_prefetch[1] = m_cells->faceNeighbors[l_cell][1];
      l_faceNeighbors_prefetch[2] = m_cells->faceNeighbors[l_cell][2];
      l_faceNeighbors_prefetch[3] = m_cells->faceNeighbors[l_cell][3];
    }
#else
    l_faceNeighbors_prefetch[0] = m_cells->faceNeighbors[l_cell][1];
    l_faceNeighbors_prefetch[1] = m_cells->faceNeighbors[l_cell][2];
    l_faceNeighbors_prefetch[2] = m_cells->faceNeighbors[l_cell][3];
     if (l_cell < (m_cells->numberOfCells-1) ) {
       l_faceNeighbors_prefetch[3] = m_cells->faceNeighbors[l_cell+1][0];
     } else {
       l_faceNeighbors_prefetch[3] = m_cells->faceNeighbors[l_cell][3];
     }
#endif
#if 1
    int l_face = 1;
    l_fluxMatricies_prefetch[0] = l_globalData->fluxMatrices[4+(l_face*12)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][0]*3)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][1])];
    l_face = 2;
    l_fluxMatricies_prefetch[1] = l_globalData->fluxMatrices[4+(l_face*12)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][0]*3)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][1])];
    l_face = 3;
    l_fluxMatricies_prefetch[2] = l_globalData->fluxMatrices[4+(l_face*12)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][0]*3)
                                                             +(m_cellInformation[l_cell].faceRelations[l_face][1])];
    l_face = 0;
    if (l_cell < (m_cells->numberOfCells-1) ) {
      l_fluxMatricies_prefetch[3] = l_globalData->fluxMatrices[4+(l_face*12)
                                                               +(m_cellInformation[l_cell+1].faceRelations[l_face][0]*3)
                                                               +(m_cellInformation[l_cell+1].faceRelations[l_face][1])];
    } else {
      l_fluxMatricies_prefetch[3] = l_globalData->fluxMatrices[4+(l_face*12)
                                                               +(m_cellInformation[l_cell].faceRelations[l_face][0]*3)
                                                               +(m_cellInformation[l_cell].faceRelations[l_face][1])];
    }
#else
    l_fluxMatricies_prefetch[0] = NULL;
    l_fluxMatricies_prefetch[1] = NULL;
    l_fluxMatricies_prefetch[2] = NULL;
    l_fluxMatricies_prefetch[3] = NULL;
#endif
#else
    l_faceNeighbors_prefetch[0] = NULL;
    l_faceNeighbors_prefetch[1] = NULL;
    l_faceNeighbors_prefetch[2] = NULL;
    l_faceNeighbors_prefetch[3] = NULL;

    l_fluxMatricies_prefetch[0] = NULL;
    l_fluxMatricies_prefetch[1] = NULL;
    l_fluxMatricies_prefetch[2] = NULL;
    l_fluxMatricies_prefetch[3] = NULL;
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

