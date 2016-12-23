/*
Copyright (c) 2015-2017, Intel Corporation

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

typedef struct seissol_flops {
  double d_nonZeroFlops;
  double d_hardwareFlops;
} seissol_flops;

seissol_flops flops_ader_actual(unsigned int i_timesteps) {
  seissol_flops ret;
  ret.d_nonZeroFlops = 0.0;
  ret.d_hardwareFlops = 0.0;

  // iterate over cells
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    unsigned int l_nonZeroFlops, l_hardwareFlops;
    // get flops
    m_timeKernel.flopsAder( l_nonZeroFlops, l_hardwareFlops );
    ret.d_nonZeroFlops  += (double)l_nonZeroFlops;
    ret.d_hardwareFlops += (double)l_hardwareFlops;
  }

  ret.d_nonZeroFlops *= (double)i_timesteps;
  ret.d_hardwareFlops *= (double)i_timesteps;

  return ret;
}

seissol_flops flops_vol_actual(unsigned int i_timesteps) {
  seissol_flops ret;
  ret.d_nonZeroFlops = 0.0;
  ret.d_hardwareFlops = 0.0;

  // iterate over cells
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    unsigned int l_nonZeroFlops, l_hardwareFlops;
    // get flops
    m_volumeKernel.flopsIntegral( l_nonZeroFlops, l_hardwareFlops );
    ret.d_nonZeroFlops  += (double)l_nonZeroFlops;
    ret.d_hardwareFlops += (double)l_hardwareFlops;
  }

  ret.d_nonZeroFlops *= (double)i_timesteps;
  ret.d_hardwareFlops *= (double)i_timesteps;

  return ret;
}

seissol_flops flops_bndlocal_actual(unsigned int i_timesteps) {
  seissol_flops ret;
  ret.d_nonZeroFlops = 0.0;
  ret.d_hardwareFlops = 0.0;

  // iterate over cells
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    unsigned int l_nonZeroFlops, l_hardwareFlops;
    // get flops
    m_boundaryKernel.flopsLocalIntegral( m_cellInformation[l_cell].faceTypes, l_nonZeroFlops, l_hardwareFlops );
    ret.d_nonZeroFlops  += (double)l_nonZeroFlops;
    ret.d_hardwareFlops += (double)l_hardwareFlops;
  }

  ret.d_nonZeroFlops *= (double)i_timesteps;
  ret.d_hardwareFlops *= (double)i_timesteps;

  return ret;
}

seissol_flops flops_bndneigh_actual(unsigned int i_timesteps) {
  seissol_flops ret;
  ret.d_nonZeroFlops = 0.0;
  ret.d_hardwareFlops = 0.0;

  // iterate over cells
  for( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
    unsigned int l_nonZeroFlops, l_hardwareFlops;
    // get flops
    m_boundaryKernel.flopsNeighborsIntegral( m_cellInformation[l_cell].faceTypes, m_cellInformation[l_cell].faceRelations, l_nonZeroFlops, l_hardwareFlops );
    ret.d_nonZeroFlops  += (double)l_nonZeroFlops;
    ret.d_hardwareFlops += (double)l_hardwareFlops;
  }

  ret.d_nonZeroFlops *= (double)i_timesteps;
  ret.d_hardwareFlops *= (double)i_timesteps;

  return ret;
}

double flops_ader(unsigned int i_timesteps) {
  double d_elems = (double)m_cells->numberOfCells;
  double d_timesteps = (double)i_timesteps;
  double flops = 0;

  for (unsigned int o = CONVERGENCE_ORDER; o > 1; o--) {
    // stiffness
    flops += 6.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( o-1 ) * (double)seissol::kernels::getNumberOfBasisFunctions( o ) * (double)NUMBER_OF_QUANTITIES;
    // star
    flops += 6.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( o-1 ) * (double)STAR_NNZ;
    // integration
    flops += 2.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( o-1 ) * (double)NUMBER_OF_QUANTITIES;
  }

  flops *= d_elems;
  flops *= d_timesteps;

  return flops;
}

double flops_vol(unsigned int i_timesteps) {
  double d_elems = (double)m_cells->numberOfCells;
  double d_timesteps = (double)i_timesteps;
  double flops = 0;

  // stiffness
  flops += 6.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)seissol::kernels::getNumberOfBasisFunctions( CONVERGENCE_ORDER-1 ) * (double)NUMBER_OF_QUANTITIES;

  // star
  flops += 6.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)STAR_NNZ;

  flops *= d_elems;
  flops *= d_timesteps;

  return flops;
}

double flops_bndlocal(unsigned int i_timesteps) {
  double d_elems = (double)m_cells->numberOfCells;
  double d_timesteps = (double)i_timesteps;
  double flops = 0;

  // flux
  flops += 8.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)seissol::kernels::getNumberOfBasisFunctions( CONVERGENCE_ORDER ) * (double)NUMBER_OF_QUANTITIES;

  // flux solver
  flops += 8.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)NUMBER_OF_QUANTITIES * (double)NUMBER_OF_QUANTITIES;

  flops *= d_elems;
  flops *= d_timesteps;

  return flops;
}

double flops_bndneigh(unsigned int i_timesteps) {
  double d_elems = (double)m_cells->numberOfCells;
  double d_timesteps = (double)i_timesteps;
  double flops = 0;

  // flux
  flops += 8.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)seissol::kernels::getNumberOfBasisFunctions( CONVERGENCE_ORDER ) * (double)NUMBER_OF_QUANTITIES;

  // flux solver
  flops += 8.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)NUMBER_OF_QUANTITIES * (double)NUMBER_OF_QUANTITIES;

  flops *= d_elems;
  flops *= d_timesteps;

  return flops;
}

double flops_local(unsigned int i_timesteps) {
  return (flops_ader(i_timesteps) + flops_vol(i_timesteps) + flops_bndlocal(i_timesteps));
}

double flops_all(unsigned int i_timesteps) {
  return (flops_local(i_timesteps) + flops_bndneigh(i_timesteps));
}

seissol_flops flops_local_actual(unsigned int i_timesteps) {
  seissol_flops ret;
  seissol_flops tmp;

  tmp = flops_ader_actual(i_timesteps);
  ret.d_nonZeroFlops = tmp.d_nonZeroFlops;
  ret.d_hardwareFlops = tmp.d_hardwareFlops;

  tmp = flops_vol_actual(i_timesteps);
  ret.d_nonZeroFlops += tmp.d_nonZeroFlops;
  ret.d_hardwareFlops += tmp.d_hardwareFlops;

  tmp = flops_bndlocal_actual(i_timesteps);
  ret.d_nonZeroFlops += tmp.d_nonZeroFlops;
  ret.d_hardwareFlops += tmp.d_hardwareFlops;

  return ret;
}

seissol_flops flops_all_actual(unsigned int i_timesteps) {
  seissol_flops ret;
  seissol_flops tmp;

  tmp = flops_local_actual(i_timesteps);
  ret.d_nonZeroFlops = tmp.d_nonZeroFlops;
  ret.d_hardwareFlops = tmp.d_hardwareFlops;

  tmp = flops_bndneigh_actual(i_timesteps);
  ret.d_nonZeroFlops += tmp.d_nonZeroFlops;
  ret.d_hardwareFlops += tmp.d_hardwareFlops;

  return ret;
}


