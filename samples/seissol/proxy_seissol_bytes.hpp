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

