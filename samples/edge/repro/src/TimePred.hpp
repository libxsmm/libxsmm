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
 * Time predictions through the ADER scheme for the elastic wave equations.
 **/
#ifndef EDGE_SEISMIC_TIME_PRED_HPP
#define EDGE_SEISMIC_TIME_PRED_HPP

#include "constants.hpp"

#if defined PP_T_KERNELS_VANILLA
#include "data/MmVanilla.hpp"
#elif defined PP_T_KERNELS_XSMM_DENSE_SINGLE
#include "data/MmXsmmSingle.hpp"
#else
#if 0
#include "data/MmXsmmFused.hpp"
#endif
#include "MmXsmmFused.hpp"
#endif

namespace edge {
  namespace elastic {
    namespace solvers {
      template< t_entityType   TL_T_EL,
                unsigned short TL_N_QTS,
                unsigned short TL_O_SP,
                unsigned short TL_O_TI,
                unsigned short TL_N_CRS >
      class TimePred;
    }
  }
}

/**
 * ADER-related functions:
 *   1) Computation of time predictions (time derivatives and time integrated DOFs) through the Cauchy–Kowalevski procedure.
 *   2) Evaluation of time prediction at specific points in time.
 *
 * @paramt TL_T_EL element type.
 * @paramt TL_N_QTS number of quantities.
 * @paramt TL_O_SP order in space.
 * @paramt TL_O_TI order in time.
 * @paramt TL_N_CRS number of concurrent forward runs (fused simulations)
 **/
template< t_entityType   TL_T_EL,
          unsigned short TL_N_QTS,
          unsigned short TL_O_SP,
          unsigned short TL_O_TI,
          unsigned short TL_N_CRS >
class edge::elastic::solvers::TimePred {
  private:
    // assemble derived template parameters
    //! dimension of the element
    static unsigned short const TL_N_DIM = C_ENT[TL_T_EL].N_DIM;
    //! number of element modes
    static unsigned short const TL_N_MDS = CE_N_ELEMENT_MODES( TL_T_EL, TL_O_SP );

    /**
     * Sets the given matrix to zero.
     *
     * @param o_mat matrix which will be set to 0.
     *
     * @paramt floating point precision of the matrix.
     **/
    template < typename TL_T_REAL >
    static void inline zero( TL_T_REAL o_mat[TL_N_QTS][TL_N_MDS][TL_N_CRS] ) {
      // reset result to zero
      for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
        for( int_md l_md = 0; l_md < TL_N_MDS; l_md++ ) {
          for( int_cfr l_cfr = 0; l_cfr < TL_N_CRS; l_cfr++ ) {
            o_mat[l_qt][l_md][l_cfr] = 0;
          }
        }
      }
    }

  public:

#if defined PP_T_KERNELS_VANILLA
    /**
     * Applies the Cauchy–Kowalevski procedure (vanilla implementation) and computes time derivatives and time integrated DOFs.
     *
     * @param i_dT time step.
     * @param i_stiffT transposed stiffness matrix (multiplied with inverse mass matrix).
     * @param i_star star matrices.
     * @param i_dofs DOFs.
     * @param i_mm vanilla matrix-matrix multiplication kernels.
     * @param o_scratch will be used as scratch memory.
     * @param o_der will be set to time derivatives.
     * @param o_tInt will be set to time integrated DOFs.
     *
     * @paramt TL_T_REAL floating point type.
     **/
    template< typename TL_T_REAL >
    static void inline ck( TL_T_REAL                            i_dT,
                           TL_T_REAL                    const   i_stiffT[TL_N_DIM][TL_N_MDS][TL_N_MDS],
                           TL_T_REAL                    const   i_star[TL_N_DIM][TL_N_QTS][TL_N_QTS],
                           TL_T_REAL                    const   i_dofs[TL_N_QTS][TL_N_MDS][TL_N_CRS],
                           data::MmVanilla< TL_T_REAL > const & i_mm,
                           TL_T_REAL                            o_scratch[TL_N_QTS][TL_N_MDS][TL_N_CRS],
                           TL_T_REAL                            o_der[TL_O_TI][TL_N_QTS][TL_N_MDS][TL_N_CRS],
                           TL_T_REAL                            o_tInt[TL_N_QTS][TL_N_MDS][TL_N_CRS] ) {
      // scalar for the time integration
      TL_T_REAL l_scalar = i_dT;

      // initialize zero-derivative, reset time integrated dofs
      for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
        for( int_md l_md = 0; l_md < TL_N_MDS; l_md++ ) {
          for( int_cfr l_cfr = 0; l_cfr < TL_N_CRS; l_cfr++ ) {
            o_der[0][l_qt][l_md][l_cfr] = i_dofs[l_qt][l_md][l_cfr];
            o_tInt[l_qt][l_md][l_cfr]   = l_scalar * i_dofs[l_qt][l_md][l_cfr];
          }
        }
      }

      // iterate over time derivatives
      for( unsigned int l_de = 1; l_de < TL_O_TI; l_de++ ) {
        // reset this derivative
        for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
          for( int_md l_md = 0; l_md < TL_N_MDS; l_md++ )
            for( int_cfr l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) o_der[l_de][l_qt][l_md][l_cr] = 0;

        // compute the derivatives
        for( unsigned short l_di = 0; l_di < TL_N_DIM; l_di++ ) {
          // multiply with transposed stiffness matrices and inverse mass matrix
          i_mm.m_kernels[(l_de-1)*2]( o_der[l_de-1][0][0],
                                      i_stiffT[l_di][0],
                                      o_scratch[0][0] );
          // multiply with star matrices
          i_mm.m_kernels[((l_de-1)*2)+1]( i_star[l_di][0],
                                          o_scratch[0][0],
                                          o_der[l_de][0][0] );
        }

        // update scalar
        l_scalar *= -i_dT / (l_de+1);

        // update time integrated dofs
        for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
          for( int_md l_md = 0; l_md < CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de ); l_md++ ) {
            for( int_cfr l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
              o_tInt[l_qt][l_md][l_cr] += l_scalar * o_der[l_de][l_qt][l_md][l_cr];
          }
        }
      }
    }
#endif

#if defined PP_T_KERNELS_XSMM
    /**
     * Applies the Cauchy–Kowalevski procedure (fused LIBXSMM version) and computes time derivatives and time integrated DOFs.
     *
     * @param i_dT time step.
     * @param i_stiffT transposed stiffness matrix (multiplied with inverse mass matrix).
     * @param i_star star matrices.
     * @param i_dofs DOFs.
     * @param i_mm matrix-matrix multiplication kernels LIBXSMM kernels.
     * @param o_scratch will be used as scratch memory.
     * @param o_der will be set to time derivatives.
     * @param o_tInt will be set to time integrated DOFs.
     *
     * @paramt TL_T_REAL floating point type.
     **/
    template< typename TL_T_REAL >
    static void inline ck( TL_T_REAL                                    i_dT,
                           TL_T_REAL                      const * const i_stiffT[(TL_O_TI>1) ? (TL_O_TI-1)*TL_N_DIM : TL_N_DIM],
                           TL_T_REAL                      const         i_star[TL_N_DIM][(TL_N_DIM==2) ? 10 : 24],
                           TL_T_REAL                      const         i_dofs[TL_N_QTS][TL_N_MDS][TL_N_CRS],
                           data::MmXsmmFused< TL_T_REAL > const &       i_mm,
                           TL_T_REAL                                    o_scratch[TL_N_QTS][TL_N_MDS][TL_N_CRS],
                           TL_T_REAL                                    o_der[TL_O_TI][TL_N_QTS][TL_N_MDS][TL_N_CRS],
                           TL_T_REAL                                    o_tInt[TL_N_QTS][TL_N_MDS][TL_N_CRS] ) {
      // scalar for the time integration
      TL_T_REAL l_scalar = i_dT;

      // initialize zero-derivative, reset time integrated dofs
      for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
        for( int_md l_md = 0; l_md < TL_N_MDS; l_md++ ) {
#pragma omp simd
          for( int_cfr l_cfr = 0; l_cfr < TL_N_CRS; l_cfr++ ) {
            o_der[0][l_qt][l_md][l_cfr] = i_dofs[l_qt][l_md][l_cfr];
            o_tInt[l_qt][l_md][l_cfr]   = l_scalar * i_dofs[l_qt][l_md][l_cfr];
          }
        }
      }

      // iterate over time derivatives
      for( unsigned int l_de = 1; l_de < TL_O_TI; l_de++ ) {
        // reset this derivative
        for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
          for( int_md l_md = 0; l_md < TL_N_MDS; l_md++ )
#pragma omp simd
            for( int_cfr l_cr = 0; l_cr < TL_N_CRS; l_cr++ ) o_der[l_de][l_qt][l_md][l_cr] = 0;

        // compute the derivatives
        for( unsigned short l_di = 0; l_di < TL_N_DIM; l_di++ ) {
          // reset to zero for basis in non-hierarchical storage
          if( TL_T_EL == TET4 || TL_T_EL == TRIA3 ) {}
          else {
            zero( o_scratch );
          }

          // multiply with transposed stiffness matrices and inverse mass matrix
          i_mm.m_kernels[(l_de-1)*(TL_N_DIM+1)+l_di]( o_der[l_de-1][0][0],
                                                      i_stiffT[(l_de-1)*TL_N_DIM+l_di],
                                                      o_scratch[0][0] );
          // multiply with star matrices
          i_mm.m_kernels[(l_de-1)*(TL_N_DIM+1)+TL_N_DIM]( i_star[l_di],
                                                          o_scratch[0][0],
                                                          o_der[l_de][0][0] );
        }

        // update scalar
        l_scalar *= -i_dT / (l_de+1);

        // update time integrated dofs
        for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
          for( int_md l_md = 0; l_md < CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de ); l_md++ ) {
#pragma omp simd
            for( int_cfr l_cr = 0; l_cr < TL_N_CRS; l_cr++ )
              o_tInt[l_qt][l_md][l_cr] += l_scalar * o_der[l_de][l_qt][l_md][l_cr];
          }
        }
      }
    }
#endif

#if defined PP_T_KERNELS_XSMM_DENSE_SINGLE
    /**
     * Applies the Cauchy–Kowalevski procedure (single forward run LIBXSMM version) and computes time derivatives and time integrated DOFs.
     *
     * @param i_dT time step.
     * @param i_stiffT transposed stiffness matrix (multiplied with inverse mass matrix).
     * @param i_star star matrices.
     * @param i_dofs DOFs.
     * @param i_mm matrix-matrix multiplication kernels.
     * @param o_scratch will be used as scratch memory.
     * @param o_der will be set to time derivatives.
     * @param o_tInt will be set to time integrated DOFs.
     *
     * @paramt TL_T_REAL floating point type.
     **/
    template <typename TL_T_REAL >
    static void inline ck( TL_T_REAL                               i_dT,
                           TL_T_REAL                       const   i_stiffT[TL_N_DIM][TL_N_MDS][TL_N_MDS],
                           TL_T_REAL                       const   i_star[TL_N_DIM][TL_N_QTS][TL_N_QTS],
                           TL_T_REAL                       const   i_dofs[TL_N_QTS][TL_N_MDS][1],
                           data::MmXsmmSingle< TL_T_REAL > const & i_mm,
                           TL_T_REAL                               o_scratch[TL_N_QTS][TL_N_MDS][1],
                           TL_T_REAL                               o_der[TL_O_TI][TL_N_QTS][TL_N_MDS][1],
                           TL_T_REAL                               o_tInt[TL_N_QTS][TL_N_MDS][1] ) {
      // scalar for the time integration
      TL_T_REAL l_scalar = i_dT;

      // initialize zero-derivative, reset time integrated dofs
      for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
#pragma omp simd
        for( int_md l_md = 0; l_md < TL_N_MDS; l_md++ ) {
          o_der[0][l_qt][l_md][0] = i_dofs[l_qt][l_md][0];
          o_tInt[l_qt][l_md][0]   = l_scalar * i_dofs[l_qt][l_md][0];
        }
      }

      // iterate over time derivatives
      for( unsigned int l_de = 1; l_de < TL_O_TI; l_de++ ) {
        // reset this derivative
        for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
#pragma omp simd
          for( int_md l_md = 0; l_md < TL_N_MDS; l_md++ ) o_der[l_de][l_qt][l_md][0] = 0;

        // compute the derivatives
        for( unsigned short l_di = 0; l_di < TL_N_DIM; l_di++ ) {
          // multiply with transposed stiffness matrices and inverse mass matrix
          i_mm.m_kernels[(l_de-1)*2]( i_stiffT[l_di][0],
                                      o_der[l_de-1][0][0],
                                      o_scratch[0][0] );
          // multiply with star matrices
          i_mm.m_kernels[((l_de-1)*2)+1]( o_scratch[0][0],
                                          i_star[l_di][0],
                                          o_der[l_de][0][0] );
        }

        // update scalar
        l_scalar *= -i_dT / (l_de+1);

        // update time integrated dofs
        for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
#pragma omp simd
          for( int_md l_md = 0; l_md < CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de ); l_md++ ) {
            o_tInt[l_qt][l_md][0] += l_scalar * o_der[l_de][l_qt][l_md][0];
          }
        }
      }
    }
#endif

    /**
     * Evaluates the time prediction, given by the time derivatives at the given points in time.
     * The points are relative to the time at which the time prediction was obtained.
     * Example:
     *   0    1.5  2.0  2.4 2.9 absolute time
     *   |-----|----x----x---x----------------->
     *         |   0.5  0.9 1.4 relative time (expected as input)
     *       time
     *    prediction
     *
     * @param i_nPts number of points.
     * @param i_pts relative pts in time.
     * @param i_der time prediction given through the time derivatives.
     * @param o_preDofs will be set to the predicted DOFs at the points in time.
     *
     * @paramt TL_T_REAL floating point type.
     **/
    template <typename TL_T_REAL >
    static void inline evalTimePrediction( unsigned short   i_nPts,
                                           TL_T_REAL const *i_pts,
                                           TL_T_REAL const  i_der[TL_O_TI][TL_N_QTS][TL_N_MDS][TL_N_CRS],
                                           TL_T_REAL       (*o_preDofs)[TL_N_QTS][TL_N_MDS][TL_N_CRS] ) {
      for( unsigned short l_pt = 0; l_pt < i_nPts; l_pt++ ) {
        // init dofs
        for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ )
          for( int_md l_md = 0; l_md < TL_N_MDS; l_md++ )
            for( int_cfr l_ru = 0; l_ru < TL_N_CRS; l_ru++ ) o_preDofs[l_pt][l_qt][l_md][l_ru] = i_der[0][l_qt][l_md][l_ru];

        // evaluate time derivatives at given point in time
        real_base l_scalar = 1.0;

        // iterate over derivatives
        for( unsigned short l_de = 1; l_de < TL_O_TI; l_de++ ) {
          // update scalar
          l_scalar *= -i_pts[l_pt] / l_de;

          for( int_qt l_qt = 0; l_qt < TL_N_QTS; l_qt++ ) {
            for( int_md l_md = 0; l_md < CE_N_ELEMENT_MODES_CK( TL_T_EL, TL_O_SP, l_de ); l_md++ ) {
              for( int_cfr l_ru = 0; l_ru < TL_N_CRS; l_ru++ )
                o_preDofs[l_pt][l_qt][l_md][l_ru] += l_scalar * i_der[l_de][l_qt][l_md][l_ru];
            }
          }
        }
      }
    }

};

#endif
