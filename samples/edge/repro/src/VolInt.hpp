/**
 * @file This file is part of EDGE.
 *
 * @author Alexander Breuer (anbreuer AT ucsd.edu)
 *         Alexander Heinecke (alexander.heinecke AT intel.com)
 *
 * @section LICENSE
 * Copyright (c) 2016-2017, Regents of the University of California
 * Copyright (c) 2016, Intel Corporation
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
 * Quadrature-free ADER-DG volume integration for seismic wave propagation.
 **/
#ifndef EDGE_SEISMIC_VOL_INT_HPP
#define EDGE_SEISMIC_VOL_INT_HPP
#include "constants.hpp"

namespace edge {
  namespace elastic {
    namespace solvers {
      template< t_entityType   TL_T_EL,
                unsigned short TL_N_QTS,
                unsigned short TL_O_SP,
                unsigned short TL_N_CRS >
      class VolInt;
    }
  }
}

/**
 * Quadrature-free ADER-DG volume integration for seismic wave propagation.
 *
 * @paramt TL_T_EL element type.
 * @paramt TL_N_QTS number of quantities.
 * @paramt TL_O_SP spatial order.
 * @paramt TL_N_CRS number of fused simulations.
 **/
template< t_entityType   TL_T_EL,
          unsigned short TL_N_QTS,
          unsigned short TL_O_SP,
          unsigned short TL_N_CRS >
class edge::elastic::solvers::VolInt {
  private:
    //! number of dimensions
    static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

    //! number of DG modes
    static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

    //! number of non-zeros in the jacobians
    static unsigned short const TL_N_NZJ = (N_DIM==2) ? 10 : 24;

  public:
#ifdef PP_T_KERNELS_VANILLA
    /**
     * Volume contribution using vanilla matrix-matrix multiplication kernels.
     *
     * @param i_stiff stiffness matrices (pre-computed, quadrature-free volume integration).
     * @param i_jac jacobians.
     * @param i_tDofs time integerated DG-DOFs.
     * @param i_mm matrix-matrix multiplication kernels.
     * @param io_dofs will be updated with local contribution of the element to the volume integral.
     * @param o_scratch will be used as scratch space for the computations.
     **/
    template< typename TL_T_REAL >
    static void inline apply( TL_T_REAL                    const   i_stiff[TL_N_DIS][TL_N_MDS][TL_N_MDS],
                              TL_T_REAL                    const   i_jac[TL_N_DIS][TL_N_QTS][TL_N_QTS],
                              TL_T_REAL                    const   i_tDofs[TL_N_QTS][TL_N_MDS][TL_N_CRS],
                              data::MmVanilla< TL_T_REAL > const & i_mm,
                              TL_T_REAL                            io_dofs[TL_N_QTS][TL_N_MDS][TL_N_CRS],
                              TL_T_REAL                            o_scratch[TL_N_QTS][TL_N_MDS][TL_N_CRS] ) {
      // iterate over dimensions
      for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
        // multiply with stiffness and inverse mass matrix
        i_mm.m_kernels[((TL_O_SP-1)*2)]( i_tDofs[0][0],
                                         i_stiff[l_di][0],
                                         o_scratch[0][0] );

        // multiply with star matrix
        i_mm.m_kernels[((TL_O_SP-1)*2)+1]( i_jac[l_di][0],
                                           o_scratch[0][0],
                                           io_dofs[0][0] );
      }
    }
#endif

#if defined PP_T_KERNELS_XSMM_DENSE_SINGLE
    /**
     * Volume contribution using non-fused LIBXSMM matrix-matrix multiplication kernels.
     *
     * @param i_stiff stiffness matrices (pre-computed, quadrature-free volume integration).
     * @param i_jac jacobians.
     * @param i_tDofs time integerated DG-DOFs.
     * @param i_mm matrix-matrix multiplication kernels.
     * @param io_dofs will be updated with local contribution of the element to the volume integral.
     * @param o_scratch will be used as scratch space for the computations.
     **/
    template< typename TL_T_REAL >
    static void inline apply( TL_T_REAL                       const   i_stiff[TL_N_DIS][TL_N_MDS][TL_N_MDS],
                              TL_T_REAL                       const   i_jac[TL_N_DIS][TL_N_QTS][TL_N_QTS],
                              TL_T_REAL                       const   i_tDofs[TL_N_QTS][TL_N_MDS][TL_N_CRS],
                              data::MmXsmmSingle< TL_T_REAL > const & i_mm,
                              TL_T_REAL                               io_dofs[TL_N_QTS][TL_N_MDS][TL_N_CRS],
                              TL_T_REAL                               o_scratch[TL_N_QTS][TL_N_MDS][TL_N_CRS] ) {
      // iterate over dimensions
      for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
        // multiply with stiffness and inverse mass matrix
        i_mm.m_kernels[((TL_O_SP-1)*2)]( i_stiff[l_di][0],
                                         i_tDofs[0][0],
                                         o_scratch[0][0] );

        // multiply with star matrix
        i_mm.m_kernels[((TL_O_SP-1)*2)+1]( o_scratch[0][0],
                                           i_jac[l_di][0],
                                           io_dofs[0][0] );
      }
    }
#endif

#if defined PP_T_KERNELS_XSMM
    /**
    * Volume contribution using fused LIBXSMM matrix-matrix multiplication kernels.
    *
    * @param i_stiff stiffness matrices (pre-computed, quadrature-free volume integration).
    * @param i_jac jacobians.
    * @param i_tDofs time integerated DG-DOFs.
    * @param i_mm matrix-matrix multiplication kernels.
    * @param io_dofs will be updated with local contribution of the element to the volume integral.
    * @param o_scratch will be used as scratch space for the computations.
    **/
    template< typename TL_T_REAL >
    static void inline apply( TL_T_REAL                      const *  const i_stiff[TL_N_DIS],
                              TL_T_REAL                      const          i_jac[TL_N_DIS][TL_N_NZJ],
                              TL_T_REAL                      const          i_tDofs[TL_N_QTS][TL_N_MDS][TL_N_CRS],
                              data::MmXsmmFused< TL_T_REAL > const &        i_mm,
                              TL_T_REAL                                     io_dofs[TL_N_QTS][TL_N_MDS][TL_N_CRS],
                              TL_T_REAL                                     o_scratch[TL_N_QTS][TL_N_MDS][TL_N_CRS] ) {
      // iterate over dimensions
      for( unsigned short l_di = 0; l_di < TL_N_DIS; l_di++ ) {
        // multiply with star matrix
        i_mm.m_kernels[(TL_O_SP-1)*(TL_N_DIS+1)+TL_N_DIS]( i_jac[l_di],
                                                           i_tDofs[0][0],
                                                           o_scratch[0][0] );

        // multiply with stiffness and inverse mass matrix
        i_mm.m_kernels[(TL_O_SP-1)*(TL_N_DIS+1)+l_di]( o_scratch[0][0],
                                                       i_stiff[l_di],
                                                       io_dofs[0][0] );
      }
    }
#endif


};

#endif
