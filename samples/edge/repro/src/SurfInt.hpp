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
 * Quadrature-free ADER-DG surface integration for seismic wave propagation.
 **/
#ifndef EDGE_SEISMIC_SURF_INT_HPP
#define EDGE_SEISMIC_SURF_INT_HPP
#include "constants.hpp"

namespace edge {
  namespace elastic {
    namespace solvers {
      template< t_entityType   TL_T_EL,
                unsigned short TL_N_QTS,
                unsigned short TL_O_SP,
                unsigned short TL_O_TI,
                unsigned short TL_N_CRS >
      class SurfInt;
    }
  }
}

/**
 * Quadrature-free ADER-DG surface integration for seismic wave propagation.
 *
 * @paramt TL_T_EL element type.
 * @paramt TL_N_QTS number of quantities.
 * @paramt TL_O_SP spatial order.
 * @paramt TL_O_TI temporal order.
 * @paramt TL_N_CRS number of fused simulations.
 **/
template< t_entityType   TL_T_EL,
          unsigned short TL_N_QTS,
          unsigned short TL_O_SP,
          unsigned short TL_O_TI,
          unsigned short TL_N_CRS >
class edge::elastic::solvers::SurfInt {
  private:
    //! number of dimensions
    static unsigned short const TL_N_DIS = C_ENT[TL_T_EL].N_DIM;

    //! number of vertices
    static unsigned short const TL_N_FVES = C_ENT[TL_T_EL].N_FACE_VERTICES;

    //! number of faces
    static unsigned short const TL_N_FAS = C_ENT[TL_T_EL].N_FACES;

    //! half the number of faces
    static unsigned short const TL_N_FAS_DIV2 =  TL_N_FAS / 2;

    //! number of DG face mods
    static unsigned short const TL_N_MDS_FA = CE_N_ELEMENT_MODES( C_ENT[TL_T_EL].TYPE_FACES, TL_O_SP );

    //! number of DG element modes
    static unsigned short const TL_N_MDS_EL = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

    //! number of neigboring contribution flux matrices
    static unsigned short const TL_N_FMNS = CE_N_FLUXN_MATRICES( TL_T_EL );

  public:
    /**
     * Determines the flux matrix id for neighboring contribution of the quadrature-free face integral.
     *
     * @param i_vIdElFaEl id of the vertex, matching the element's vertex 0, from the perspective of the adjacent element w.r.t. to the reference element.
     * @param i_fIdElFaEl id of the face from the perspective of the adjacent element w.r.t. to the reference element.
     **/
    static unsigned short inline fMatId( unsigned short i_vIdElFaEl,
                                         unsigned short i_fIdElFaEl ) {
      return i_vIdElFaEl*TL_N_FAS+i_fIdElFaEl;
    }

#ifdef PP_T_KERNELS_VANILLA
    /**
     * Single contribution (local or neighboring) for a face using vanilla matrix-matrix multiplication kernels.
     *
     * @param i_fInt face integration matrix (pre-computed, quadrature-free surface integration).
     * @param i_fSol flux solver.
     * @param i_tDofs time integrated DG-DOFs.
     * @param i_mm matrix-matrix multiplication kernels.
     * @param io_dofs will be updated with the single contribution of a face to the surface integral.
     * @param o_scratch will be used as scratch space for the computations.
     * @param i_dofsP DOFs for prefetching (not used).
     * @param i_fId flux matrix id (not used).
     *
     * @paramt TL_T_REAL floating point precision.
     **/
    template< typename TL_T_REAL >
    static void inline faS( TL_T_REAL                    const   i_fInt[TL_N_MDS_EL][TL_N_MDS_EL],
                            TL_T_REAL                    const   i_fSol[TL_N_QTS][TL_N_QTS],
                            TL_T_REAL                    const   i_tDofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                            data::MmVanilla< TL_T_REAL > const & i_mm,
                            TL_T_REAL                            io_dofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                            TL_T_REAL                            o_scratch[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                            TL_T_REAL                    const   i_dofsP[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS] = nullptr,
                            unsigned short                       i_fId = std::numeric_limits< unsigned short >::max() ) {
      EDGE_LOG_FATAL << "TODO";
      // multiply with face integration matrix
      i_mm.m_kernels[((TL_O_TI-1)*2)+2]( i_tDofs[0][0],
                                         i_fInt[0],
                                         o_scratch[0][0] );
      // multiply with flux solver
      i_mm.m_kernels[((TL_O_TI-1)*2)+3]( i_fSol[0],
                                         o_scratch[0][0],
                                         io_dofs[0][0] );
    }
#endif

#if defined PP_T_KERNELS_XSMM_DENSE_SINGLE
    /**
     * Single contribution (local or neighboring) for a face using non-fused LIBXSMM matrix-matrix multiplication kernels.
     *
     * @param i_fInt face integration matrix (pre-computed, quadrature-free surface integration).
     * @param i_fSol flux solver.
     * @param i_tDofs time integrated DG-DOFs.
     * @param i_mm matrix-matrix multiplication kernels.
     * @param io_dofs will be updated with the single contribution of a face to the surface integral.
     * @param o_scratch will be used as scratch space for the computations.
     * @param i_dofsP DOFs for prefetching (not used).
     * @param i_fId flux matrix id (not used).
     *
     * @paramt TL_T_REAL floating point precision.
     **/
    template< typename TL_T_REAL >
    static void inline faS( TL_T_REAL                       const   i_fInt[TL_N_MDS_EL][TL_N_MDS_EL],
                            TL_T_REAL                       const   i_fSol[TL_N_QTS][TL_N_QTS],
                            TL_T_REAL                       const   i_tDofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                            data::MmXsmmSingle< TL_T_REAL > const & i_mm,
                            TL_T_REAL                               io_dofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                            TL_T_REAL                               o_scratch[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                            TL_T_REAL                       const   i_dofsP[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS] = nullptr,
                            unsigned short                          i_fId = std::numeric_limits< unsigned short >::max() ) {
      EDGE_LOG_FATAL << "TODO";
      // multiply with flux matrix
      i_mm.m_kernels[((TL_O_TI-1)*2)+3]( i_fInt[0],
                                         i_tDofs[0][0],
                                         o_scratch[0][0] );

      // multiply with flux solver
      i_mm.m_kernels[((TL_O_TI-1)*2)+4]( o_scratch[0][0],
                                         i_fSol[0],
                                         io_dofs[0][0] );
    }
#endif


#ifdef PP_T_KERNELS_VANILLA
    /**
     * Element local contribution using vanilla matrix-matrix multiplication kernels.
     *
     * @param i_fIntL local face integration matrices (pre-computed, quadrature-free surface integration).
     * @param i_fIntT transposed face integration matrices (pre-computed, quadrature-free surface integration).
     * @param i_fSol flux solvers.
     * @param i_tDofs time integerated DG-DOFs.
     * @param i_mm matrix-matrix multiplication kernels.
     * @param io_dofs will be updated with local contribution of the element to the surface integral.
     * @param o_scratch will be used as scratch space for the computations.
     * @param i_dofsP DOFs for prefetching (not used).
     * @param i_tDofsP time integrated DOFs for prefetching (not used).
     *
     * @paramt TL_T_REAL floating point precision.
     **/
    template< typename TL_T_REAL >
    static void inline local( TL_T_REAL                    const   i_fIntL[TL_N_FAS][TL_N_MDS_EL][TL_N_MDS_FA],
                              TL_T_REAL                    const   i_fIntT[TL_N_FAS][TL_N_MDS_FA][TL_N_MDS_EL],
                              TL_T_REAL                    const   i_fSol[TL_N_FAS][TL_N_QTS][TL_N_QTS],
                              TL_T_REAL                    const   i_tDofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              data::MmVanilla< TL_T_REAL > const & i_mm,
                              TL_T_REAL                            io_dofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              TL_T_REAL                            o_scratch[2][TL_N_QTS][TL_N_MDS_FA][TL_N_CRS],
                              TL_T_REAL                    const   i_dofsP[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS] = nullptr,
                              TL_T_REAL                    const   i_tDofsP[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS] = nullptr ) {
      // iterate over faces
      for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
        // multiply with first face integration matrix
        i_mm.m_kernels[((TL_O_TI-1)*2)+2]( i_tDofs[0][0],
                                           i_fIntL[l_fa][0],
                                           o_scratch[0][0][0] );

        // multiply with flux solver
        i_mm.m_kernels[((TL_O_TI-1)*2)+3]( i_fSol[l_fa][0],
                                           o_scratch[0][0][0],
                                           o_scratch[1][0][0] );

        // multiply with second face integration matrix
        i_mm.m_kernels[((TL_O_TI-1)*2)+4]( o_scratch[1][0][0],
                                           i_fIntT[l_fa][0],
                                           io_dofs[0][0] );
      }
    }
#endif

#if defined PP_T_KERNELS_XSMM_DENSE_SINGLE
    /**
     * Element local contribution using non-fused LIBXSMM matrix-matrix multiplication kernels.
     *
     * @param i_fIntL local face integration matrices (pre-computed, quadrature-free surface integration).
     * @param i_fIntT transposed face integration matrices (pre-computed, quadrature-free surface integration).
     * @param i_fSol flux solvers.
     * @param i_tDofs time integerated DG-DOFs.
     * @param i_mm matrix-matrix multiplication kernels.
     * @param io_dofs will be updated with local contribution of the element to the surface integral.
     * @param o_scratch will be used as scratch space for the computations.
     * @param i_dofsP DOFs for prefetching (not used).
     * @param i_tDofsP time integrated DOFs for prefetching (not used).
     *
     * @paramt TL_T_REAL floating point precision.
     **/
    template< typename TL_T_REAL >
    static void inline local( TL_T_REAL                       const   i_fIntL[TL_N_FAS][TL_N_MDS_EL][TL_N_MDS_FA],
                              TL_T_REAL                       const   i_fIntT[TL_N_FAS][TL_N_MDS_FA][TL_N_MDS_EL],
                              TL_T_REAL                       const   i_fSol[TL_N_FAS][TL_N_QTS][TL_N_QTS],
                              TL_T_REAL                       const   i_tDofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              data::MmXsmmSingle< TL_T_REAL > const & i_mm,
                              TL_T_REAL                               io_dofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              TL_T_REAL                               o_scratch[2][TL_N_QTS][TL_N_MDS_FA][TL_N_CRS],
                              TL_T_REAL                       const   i_dofsP[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              TL_T_REAL                       const   i_tDofsP[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS] ) {
      // iterate over faces
      for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
        // multiply with first face integration matrix
        i_mm.m_kernels[((TL_O_TI-1)*2)+2]( i_fIntL[l_fa][0],
                                           i_tDofs[0][0],
                                           o_scratch[0][0][0],
                                           nullptr,
                                           i_dofsP[0][0],
                                           nullptr );

        // multiply with flux solver
        i_mm.m_kernels[((TL_O_TI-1)*2)+3]( o_scratch[0][0][0],
                                           i_fSol[l_fa][0],
                                           o_scratch[1][0][0] );

        // multiply with second face integration matrix
        i_mm.m_kernels[((TL_O_TI-1)*2)+4]( i_fIntT[l_fa][0],
                                           o_scratch[1][0][0],
                                           io_dofs[0][0],
                                           nullptr,
                                           i_tDofsP[0][0],
                                           nullptr );
      }
    }
#endif

#if defined PP_T_KERNELS_XSMM
    /**
    * Element local contribution using fused LIBXSMM matrix-matrix multiplication kernels.
    *
    * @param i_fIntL local face integration matrices (pre-computed, quadrature-free surface integration).
    * @param i_fIntT transposed face integration matrices (pre-computed, quadrature-free surface integration).
    * @param i_fSol flux solvers.
    * @param i_tDofs time integerated DG-DOFs.
    * @param i_mm matrix-matrix multiplication kernels.
    * @param io_dofs will be updated with local contribution of the element to the surface integral.
    * @param o_scratch will be used as scratch space for the computations.
    * @param i_dofsP DOFs for prefetching (not used).
    * @param i_tDofsP time integrated DOFs for prefetching (not used).
    *
    * @paramt TL_T_REAL floating point precision.
    **/
    template< typename TL_T_REAL >
    static void inline local( TL_T_REAL                      const *  const i_fIntL[TL_N_FAS],
                              TL_T_REAL                      const *  const i_fIntT[TL_N_FAS],
                              TL_T_REAL                      const          i_fSol[TL_N_FAS][TL_N_QTS][TL_N_QTS],
                              TL_T_REAL                      const          i_tDofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              data::MmXsmmFused< TL_T_REAL > const &        i_mm,
                              TL_T_REAL                                     io_dofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              TL_T_REAL                                     o_scratch[2][TL_N_QTS][TL_N_MDS_FA][TL_N_CRS],
                              TL_T_REAL                      const          i_dofsP[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS] = nullptr,
                              TL_T_REAL                      const          i_tDofsP[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS] = nullptr ) {
      // iterate over faces
      for( unsigned short l_fa = 0; l_fa < TL_N_FAS; l_fa++ ) {
        // local flux matrix
        i_mm.m_kernels[TL_O_TI*(TL_N_DIS+1)+l_fa]( i_tDofs[0][0],
                                                   i_fIntL[l_fa],
                                                   o_scratch[0][0][0] );

        // flux solver
        i_mm.m_kernels[TL_O_TI*(TL_N_DIS+1)+TL_N_FAS+TL_N_FMNS+TL_N_FAS]( i_fSol[l_fa][0],
                                                                          o_scratch[0][0][0],
                                                                          o_scratch[1][0][0],
                                                                          nullptr,
                                                                          (l_fa < TL_N_FAS_DIV2) ? i_dofsP[0][0] : i_tDofsP[0][0],
                                                                          nullptr );

        // transposed flux matrix
        i_mm.m_kernels[TL_O_TI*(TL_N_DIS+1)+TL_N_FAS+TL_N_FMNS+l_fa]( o_scratch[1][0][0],
                                                                      i_fIntT[l_fa],
                                                                      io_dofs[0][0] );
      }
    }
#endif


#ifdef PP_T_KERNELS_VANILLA
    /**
     * Neighboring contribution of a single adjacent element using vanilla matrix-matrix multiplication kernels.
     *
     * @param i_fIntLN local or neighboring face integration matrix (pre-computed, quadrature-free surface integration).
     * @param i_fIntT transposed face integration matrices (pre-computed, quadrature-free surface integration).     * @param i_fSol flux solvers.
     * @param i_tDofs time integerated DG-DOFs.
     * @param i_mm matrix-matrix multiplication kernels.
     * @param io_dofs will be updated with the contribution of the adjacent element to the surface integral.
     * @param o_scratch will be used as scratch space for the computations.
     * @param i_pre DOFs or tDOFs for prefetching (not used).
     * @param i_fa local face (not used).
     * @param i_fId flux matrix id (not used).
     *
     * @paramt TL_T_REAL floating point precision.
     **/
    template< typename TL_T_REAL >
    static void inline neigh( TL_T_REAL                    const   i_fIntLN[TL_N_MDS_EL][TL_N_MDS_FA],
                              TL_T_REAL                    const   i_fIntT[TL_N_MDS_FA][TL_N_MDS_EL],
                              TL_T_REAL                    const   i_fSol[TL_N_QTS][TL_N_QTS],
                              TL_T_REAL                    const   i_tDofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              data::MmVanilla< TL_T_REAL > const & i_mm,
                              TL_T_REAL                            io_dofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              TL_T_REAL                            o_scratch[2][TL_N_QTS][TL_N_MDS_FA][TL_N_CRS],
                              TL_T_REAL                    const   i_pre[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS] = nullptr,
                              unsigned short                       i_fa = std::numeric_limits< unsigned short >::max(),
                              unsigned short                       i_fId = std::numeric_limits< unsigned short >::max() ) {
      // multiply with first face integration matrix
      i_mm.m_kernels[((TL_O_TI-1)*2)+2]( i_tDofs[0][0],
                                         i_fIntLN[0],
                                         o_scratch[0][0][0] );

      // multiply with flux solver
      i_mm.m_kernels[((TL_O_TI-1)*2)+3]( i_fSol[0],
                                         o_scratch[0][0][0],
                                         o_scratch[1][0][0] );

      // multiply with second face integration matrix
      i_mm.m_kernels[((TL_O_TI-1)*2)+4]( o_scratch[1][0][0],
                                         i_fIntT[0],
                                         io_dofs[0][0] );
    }
#endif

#ifdef PP_T_KERNELS_XSMM_DENSE_SINGLE
    /**
     * Neighboring contribution of a single adjacent element using non-fused LIBXSMM matrix-matrix multiplication kernels.
     *
     * @param i_fIntLN local or neighboring face integration matrix (pre-computed, quadrature-free surface integration).
     * @param i_fIntT transposed face integration matrices (pre-computed, quadrature-free surface integration).
     * @param i_fSol flux solvers.
     * @param i_tDofs time integerated DG-DOFs.
     * @param i_mm matrix-matrix multiplication kernels.
     * @param io_dofs will be updated with the contribution of the adjacent element to the surface integral.
     * @param o_scratch will be used as scratch space for the computations.
     * @param i_pre DOFs or tDOFs for prefetching (not used).
     * @param i_fa local face (not used).
     * @param i_fId flux matrix id (not used).
     *
     * @paramt TL_T_REAL floating point precision.
     **/
    template< typename TL_T_REAL >
    static void inline neigh( TL_T_REAL                       const   i_fIntLN[TL_N_MDS_EL][TL_N_MDS_FA],
                              TL_T_REAL                       const   i_fIntT[TL_N_MDS_FA][TL_N_MDS_EL],
                              TL_T_REAL                       const   i_fSol[TL_N_QTS][TL_N_QTS],
                              TL_T_REAL                       const   i_tDofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              data::MmXsmmSingle< TL_T_REAL > const & i_mm,
                              TL_T_REAL                               io_dofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              TL_T_REAL                               o_scratch[2][TL_N_QTS][TL_N_MDS_FA][TL_N_CRS],
                              TL_T_REAL                       const   i_pre[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              unsigned short                          i_fa = std::numeric_limits< unsigned short >::max(),
                              unsigned short                          i_fId = std::numeric_limits< unsigned short >::max() ) {
      // multiply with first face integration matrix
      i_mm.m_kernels[((TL_O_TI-1)*2)+2]( i_fIntLN[0],
                                         i_tDofs[0][0],
                                         o_scratch[0][0][0],
                                         nullptr,
                                         i_pre[0][0],
                                         nullptr );

      // multiply with flux solver
      i_mm.m_kernels[((TL_O_TI-1)*2)+3]( o_scratch[0][0][0],
                                         i_fSol[0],
                                         o_scratch[1][0][0] );

      // multiply with second face integration matrix
      i_mm.m_kernels[((TL_O_TI-1)*2)+4]( i_fIntT[0],
                                         o_scratch[1][0][0],
                                         io_dofs[0][0],
                                         nullptr,
                                         i_pre[0][0],
                                         nullptr );
    }
#endif

#ifdef PP_T_KERNELS_XSMM
    /**
     * Neighboring contribution of a single adjacent element using fused LIBXSMM matrix-matrix multiplication kernels.
     *
     * @param i_fInt face integration matrices (pre-computed, quadrature-free surface integration).
     * @param i_fSol flux solvers.
     * @param i_tDofs time integerated DG-DOFs.
     * @param i_mm matrix-matrix multiplication kernels.
     * @param io_dofs will be updated with the contribution of the adjacent element to the surface integral.
     * @param o_scratch will be used as scratch space for the computations.
     * @param i_pre DOFs or tDOFs for prefetching.
     * @param i_fa local face.
     * @param i_fId flux matrix id of the local or neighboring flux matrix (+ TL_N_FAS if neighboring).
     *
     * @paramt TL_T_REAL floating point precision.
     **/
    template< typename TL_T_REAL >
    static void inline neigh( TL_T_REAL                      const  * i_fIntLN,
                              TL_T_REAL                      const  * i_fIntT,
                              TL_T_REAL                      const    i_fSol[TL_N_QTS][TL_N_QTS],
                              TL_T_REAL                      const    i_tDofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              data::MmXsmmFused< TL_T_REAL > const  & i_mm,
                              TL_T_REAL                               io_dofs[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              TL_T_REAL                               o_scratch[2][TL_N_QTS][TL_N_MDS_FA][TL_N_CRS],
                              TL_T_REAL                      const    i_pre[TL_N_QTS][TL_N_MDS_EL][TL_N_CRS],
                              unsigned short                          i_fa,
                              unsigned short                          i_fId ) {
      // local or neighboring flux matrix
      i_mm.m_kernels[TL_O_TI*(TL_N_DIS+1)+i_fId]( i_tDofs[0][0],
                                                  i_fIntLN,
                                                  o_scratch[0][0][0] );

      // flux solver
      i_mm.m_kernels[TL_O_TI*(TL_N_DIS+1)+TL_N_FAS+TL_N_FMNS+TL_N_FAS]( i_fSol[0],
                                                                        o_scratch[0][0][0],
                                                                        o_scratch[1][0][0],
                                                                        nullptr,
                                                                        i_pre[0][0],
                                                                        nullptr );

      // transposed flux matrix
      i_mm.m_kernels[TL_O_TI*(TL_N_DIS+1)+TL_N_FAS+TL_N_FMNS+i_fa]( o_scratch[1][0][0],
                                                                    i_fIntT,
                                                                    io_dofs[0][0] );
    }
#endif

};

#endif
