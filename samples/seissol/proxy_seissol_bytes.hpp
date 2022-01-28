/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#ifndef PROXY_SEISSOL_BYTES_HPP
#define PROXY_SEISSOL_BYTES_HPP

double bytes_ader(unsigned int i_timesteps) {
  double d_elems = (double)m_cells->numberOfCells;
  double d_timesteps = (double)i_timesteps;
  double bytes = 0;

  // DOFs load and tDOFs write
#ifdef __USE_DERS
  for (int o  = CONVERGENCE_ORDER; o > 0; o--) {
    bytes += (double)sizeof(real) * 1.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( o ) * (double)NUMBER_OF_QUANTITIES;
  }
#endif
  bytes += (double)sizeof(real) * 3.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)NUMBER_OF_QUANTITIES;

  // star
  bytes += (double)sizeof(real) * 3.0 * (double)STAR_NNZ;

  bytes *= d_elems;
  bytes *= d_timesteps;

  return bytes;
}

double bytes_vol(unsigned int i_timesteps) {
  double d_elems = (double)m_cells->numberOfCells;
  double d_timesteps = (double)i_timesteps;
  double bytes = 0;

  // tDOFs load, DOFs write
  bytes += (double)sizeof(real) * 3.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)NUMBER_OF_QUANTITIES;
  // star
  bytes += (double)sizeof(real) * 3.0 * (double)STAR_NNZ;

  bytes *= d_elems;
  bytes *= d_timesteps;

  return bytes;
}

double bytes_bndlocal(unsigned int i_timesteps) {
  double d_elems = (double)m_cells->numberOfCells;
  double d_timesteps = (double)i_timesteps;
  double bytes = 0;

  // tDOFs load, DOFs write
  bytes += (double)sizeof(real) * 3.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)NUMBER_OF_QUANTITIES;
  // flux
  bytes += (double)sizeof(real) * 4.0 * (double)NUMBER_OF_QUANTITIES * (double)NUMBER_OF_QUANTITIES;

  bytes *= d_elems;
  bytes *= d_timesteps;

  return bytes;
}

double bytes_local(unsigned int i_timesteps) {
  double d_elems = (double)m_cells->numberOfCells;
  double d_timesteps = (double)i_timesteps;
  double bytes = 0;

  // DOFs load, tDOFs sum of ader, vol, boundary
  bytes += (double)sizeof(real) * 3.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)NUMBER_OF_QUANTITIES;
  // star
  bytes += (double)sizeof(real) * 3.0 * (double)STAR_NNZ;
  // flux solver
  bytes += (double)sizeof(real) * 4.0 * (double)NUMBER_OF_QUANTITIES * (double)NUMBER_OF_QUANTITIES;

  bytes *= d_elems;
  bytes *= d_timesteps;

  return bytes;
}

double bytes_bndneigh(unsigned int i_timesteps) {
  double d_elems = (double)m_cells->numberOfCells;
  double d_timesteps = (double)i_timesteps;
  double bytes = 0;

  // 4 tDOFs/DERs load, DOFs write
#ifdef __USE_DERS
  bytes += (double)sizeof(real) * 2.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)NUMBER_OF_QUANTITIES;
  // load neighbors' DERs
  for (int o  = CONVERGENCE_ORDER; o > 0; o--) {
    bytes += (double)sizeof(real) * 4.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( o ) * (double)NUMBER_OF_QUANTITIES;
  }
#else
  bytes += (double)sizeof(real) * 6.0 * (double)seissol::kernels::getNumberOfAlignedBasisFunctions( CONVERGENCE_ORDER ) * (double)NUMBER_OF_QUANTITIES;
#endif
  // flux
  bytes += (double)sizeof(real) * 4.0 * (double)NUMBER_OF_QUANTITIES * (double)NUMBER_OF_QUANTITIES;

  bytes *= d_elems;
  bytes *= d_timesteps;

  return bytes;
}

double bytes_all(unsigned int i_timesteps) {
  return (bytes_local(i_timesteps) + bytes_bndneigh(i_timesteps));
}

#endif /*PROXY_SEISSOL_BYTES_HPP*/

