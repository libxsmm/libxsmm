/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef PROXY_SEISSOL_FLOPS_HPP
#define PROXY_SEISSOL_FLOPS_HPP

typedef struct seissol_flops {
  double d_nonZeroFlops;
  double d_hardwareFlops;
} seissol_flops;

seissol_flops flops_ader_actual(unsigned int i_timesteps) {
  seissol_flops ret;
  ret.d_nonZeroFlops = 0.0;
  ret.d_hardwareFlops = 0.0;

  // iterate over cells
  for ( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
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
  for ( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
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
  for ( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
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
  for ( unsigned int l_cell = 0; l_cell < m_cells->numberOfCells; l_cell++ ) {
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

#endif /*PROXY_SEISSOL_FLOPS_HPP*/

