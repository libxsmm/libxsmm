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
 * ADER-DG solver for the elastic wave equations.
 **/
#ifndef ADER_DG_HPP
#define ADER_DG_HPP

#include <limits>
#include <cassert>
#include "constants.hpp"
#if 0
#include "mesh/common.hpp"
#include "linalg/Mappings.hpp"
#endif
#include "TimePred.hpp"
#include "VolInt.hpp"
#include "SurfInt.hpp"
#include "Receivers_dummy.hpp"
#if 0
#include "io/Receivers.h"
#include "InternalBoundary.hpp"
#include "FrictionLaws.hpp"
#endif

#if defined(PP_T_KERNELS_XSMM) || defined(PP_T_KERNELS_XSMM_DENSE_SINGLE)
#include <libxsmm.h>
#endif

namespace edge {
  namespace elastic {
    namespace solvers {
      class AderDg;
    }
  }
}

class edge::elastic::solvers::AderDg {
  //private:
  public:
    /**
     * Local step: Cauchy Kowalevski + volume.
     *
     * @param i_first first element considered.
     * @param i_nElements number of elements.
     * @param i_time time of the initial DOFs.
     * @param i_dT time step.
     * @param i_firstSpRp first sparse rupture-element.
     * @param i_firstSpRe first sparse receiver entity.
     * @param i_elFa faces adjacent to the elements.
     * @param i_faChars face characteristics.
     * @param i_elChars element characteristics.
     * @param i_dg const DG data.
     * @param i_starM star matrices.
     * @param i_fluxSolvers flux solvers for the local element's contribution.
     * @param io_dofs DOFs.
     * @param o_tInt will be set to time integrated DOFs.
     * @param o_tRup will be set to DOFs for rupture elements.
     * @param io_recvs will be updated with receiver info.
     * @param i_kernels kernels of XSMM-library for the local step (if enabled).
     *
     * @paramt TL_T_INT_LID integer type of local entity ids.
     * @paramt TL_T_REAL floating point type.
     * @paramt TL_T_MM matrix-matrix multiplication kernels.
     **/
    template < typename TL_T_INT_LID,
               typename TL_T_REAL,
               typename TL_T_MM >
    static void local( TL_T_INT_LID                     i_first,
                       TL_T_INT_LID                     i_nElements,
                       double                           i_time,
                       double                           i_dT,
                       TL_T_INT_LID                     i_firstSpRp,
                       TL_T_INT_LID                     i_firstSpRe,
                       TL_T_INT_LID            const (* i_elFa)[ C_ENT[T_SDISC.ELEMENT].N_FACES ],
                       t_faceChars             const  * i_faChars,
                       t_elementChars          const  * i_elChars,
                       t_dg                    const  & i_dg,
                       t_matStar               const (* i_starM)[N_DIM],
                       t_fluxSolver            const (* i_fluxSolvers)[ C_ENT[T_SDISC.ELEMENT].N_FACES ],
                       TL_T_REAL                     (* io_dofs)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS],
                       TL_T_REAL                     (* o_tInt)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS],
                       TL_T_REAL                     (* o_tRup)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS],
                       edge::io::Receivers            & io_recvs,
                       TL_T_MM const                  & i_mm ) {
#if __has_builtin(__builtin_assume_aligned)
      // share alignment with compiler
      (void) __builtin_assume_aligned(io_dofs, ALIGNMENT.ELEMENT_MODES.PRIVATE);
      (void) __builtin_assume_aligned(o_tInt,  ALIGNMENT.ELEMENT_MODES.PRIVATE);
#endif

      // counter of elements with faces having rupture physics
      TL_T_INT_LID l_elRp = i_firstSpRp;

      // counter for receivers
      unsigned int l_enRe = i_firstSpRe;

      // temporary data structurre for product for two-way mult and receivers
      TL_T_REAL (*l_tmpEl)[N_ELEMENT_MODES][N_CRUNS] = parallel::g_scratchMem->tRes;

      // buffer for derivatives
      TL_T_REAL (*l_derBuffer)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS] = parallel::g_scratchMem->dBuf;

      // iterate over all elements
      for( TL_T_INT_LID l_el = i_first; l_el < i_first+i_nElements; l_el++ ) {
        if( (i_elChars[l_el].spType & RUPTURE) != RUPTURE ) {}
        else {
          // save DOFs of the rupture element
          for( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
            for( unsigned short l_md = 0; l_md < N_ELEMENT_MODES; l_md++ )
              for( unsigned short l_ru = 0; l_ru < N_CRUNS; l_ru++ )
                o_tRup[l_elRp][l_qt][l_md][l_ru] = io_dofs[l_el][l_qt][l_md][l_ru];

          // increase rupture element counter
          l_elRp++;
        }

        /*
         * compute ader time integration
         */
        TimePred< T_SDISC.ELEMENT,
                  N_QUANTITIES,
                  ORDER,
                  ORDER,
                  N_CRUNS >::ck( (TL_T_REAL)  i_dT,
                                              i_dg.mat.stiffT,
                                            &(i_starM[l_el][0].mat), // TODO: fix struct
                                              io_dofs[l_el],
                                              i_mm,
                                              l_tmpEl,
                                              l_derBuffer,
                                              o_tInt[l_el] );

        /*
         * Write receivers (if required)
         */
        if( !( (i_elChars[l_el].spType & RECEIVER) == RECEIVER) ) {} // no receivers in the current element
        else { // we have receivers in the current element
          while( true ) { // iterate of possible multiple receiver-ouput per time step
            TL_T_REAL l_rePt[1];
            l_rePt[0] = io_recvs.getRecvTimeRel( l_enRe, i_time, i_dT );
            if( !(l_rePt[0] >= 0) ) break;
            else {
              // eval time prediction at the given point
              TimePred< T_SDISC.ELEMENT,
                        N_QUANTITIES,
                        ORDER,
                        ORDER,
                        N_CRUNS >::evalTimePrediction( 1,
                                                       l_rePt,
                                                       l_derBuffer,
                          (TL_T_REAL (*)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS])l_tmpEl );

              // write this time prediction
              io_recvs.writeRecvAll( l_enRe, l_tmpEl );
            }
          }
          l_enRe++;
        }

        /*
         * compute volume contribution
         */
        VolInt< T_SDISC.ELEMENT,
                N_QUANTITIES,
                ORDER,
                N_CRUNS >::apply(   i_dg.mat.stiff,
                                  & i_starM[l_el][0].mat, // TODO: fix struct
                                    o_tInt[l_el],
                                    i_mm,
                                    io_dofs[l_el],
                                    l_tmpEl );

        /*
         * prefetches for next iteration
         */
        const TL_T_REAL (* l_preDofs)[N_ELEMENT_MODES][N_CRUNS] = nullptr;
        const TL_T_REAL (* l_preTint)[N_ELEMENT_MODES][N_CRUNS] = nullptr;
        if( l_el < i_first+i_nElements-1 ) {
          l_preDofs = io_dofs[l_el+1];
          l_preTint = o_tInt[l_el+1];
        }
        else {
          l_preDofs = io_dofs[l_el];
          l_preTint = o_tInt[l_el];
        }

         /*
          * compute local surface contribution
          */
        // reuse derivative buffer
        TL_T_REAL (*l_tmpFa)[N_QUANTITIES][N_FACE_MODES][N_CRUNS] =
          (TL_T_REAL (*)[N_QUANTITIES][N_FACE_MODES][N_CRUNS]) parallel::g_scratchMem->dBuf;
        // call kernel
        SurfInt< T_SDISC.ELEMENT,
                 N_QUANTITIES,
                 ORDER,
                 ORDER,
                 N_CRUNS >::local( i_dg.mat.fluxL,
                                   i_dg.mat.fluxT,
                                   ( TL_T_REAL (*)[N_QUANTITIES][N_QUANTITIES] )  ( i_fluxSolvers[l_el][0].solver[0] ), // TODO: fix struct
                                   o_tInt[l_el],
                                   i_mm,
                                   io_dofs[l_el],
                                   l_tmpFa,
                                   l_preDofs,
                                   l_preTint );
      }
    }

    /**
     * Performs the neighboring updates of the ADER-DG scheme.
     *
     * @param i_first first element considered.
     * @param i_nElements number of elements.
     * @param i_dg constant DG data.
     * @param i_faChars face characteristics.
     * @param i_fluxSolvers flux solvers for the neighboring elements' contribution.
     * @param i_elFa elements' adjacent faces.
     * @param i_elFaEl face-neighboring elements.
     * @param i_faElSpRp adjacency information from sparse rupture faces to sparse rupture elements.
     * @param i_elFaSpRp adjacnecy information from sparse rupture elements to sparse rupture faces.
     * @param i_fIdElFaEl local face ids of face-neighboring elememts.
     * @param i_vIdElFaEl local vertex ids w.r.t. the shared face from the neighboring elements' perspsective.
     * @param i_tInt time integrated degrees of freedom.
     * @param i_updatesSpRp surface updates resulting from rupture physics.
     * @param io_dofs DOFs which will be updated with neighboring elements' contribution.
     * @param i_kernels kernels of XSMM-library for the neighboring step (if enabled).
     *
     * @paramt TL_T_INT_LID integer type of local entity ids.
     * @paramt TL_T_REAL type used for floating point arithmetic.
     * @paramt TL_T_MM type of the matrix-matrix multiplication kernels.
     **/
    template< typename TL_T_INT_LID,
              typename TL_T_REAL,
              typename TL_T_MM >
    static void neigh( TL_T_INT_LID            i_first,
                       TL_T_INT_LID            i_nElements,
                       TL_T_INT_LID            i_firstSpRp,
                       t_dg           const  & i_dg,
                       t_faceChars    const  * i_faChars,
                       t_fluxSolver   const (* i_fluxSolvers)[ C_ENT[T_SDISC.ELEMENT].N_FACES ],
                       TL_T_INT_LID   const (* i_elFa)[C_ENT[T_SDISC.ELEMENT].N_FACES],
                       TL_T_INT_LID   const (* i_elFaEl)[C_ENT[T_SDISC.ELEMENT].N_FACES],
                       TL_T_INT_LID  const  (* i_faElSpRp)[2],
                       TL_T_INT_LID  const  (* i_elFaSpRp)[C_ENT[T_SDISC.ELEMENT].N_FACES],
                       unsigned short const (* i_fIdElFaEl)[C_ENT[T_SDISC.ELEMENT].N_FACES],
                       unsigned short const (* i_vIdElFaEl)[C_ENT[T_SDISC.ELEMENT].N_FACES],
                       TL_T_REAL      const (* i_tInt)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS],
                       TL_T_REAL      const (* i_updatesSpRp)[2][N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS],
                       TL_T_REAL            (* io_dofs)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS],
                       TL_T_MM const         & i_mm ) {
#if __has_builtin(__builtin_assume_aligned)
      // share alignment with compiler
      (void) __builtin_assume_aligned(i_tInt,  ALIGNMENT.ELEMENT_MODES.PRIVATE);
      (void) __builtin_assume_aligned(io_dofs, ALIGNMENT.ELEMENT_MODES.PRIVATE);
#endif

      // temporary product for three-way mult
        TL_T_REAL (*l_tmpFa)[N_QUANTITIES][N_FACE_MODES][N_CRUNS] =
          (TL_T_REAL (*)[N_QUANTITIES][N_FACE_MODES][N_CRUNS]) parallel::g_scratchMem->dBuf;

      // iterate over elements
      for( TL_T_INT_LID l_el = i_first; l_el < i_first+i_nElements; l_el++ ) {

        // add neighboring contribution
        for( TL_T_INT_LID l_fa = 0; l_fa < C_ENT[T_SDISC.ELEMENT].N_FACES; l_fa++ ) {
          TL_T_INT_LID l_faId = i_elFa[l_el][l_fa];
          TL_T_INT_LID l_ne;

          // determine flux matrix id
          unsigned short l_fId = SurfInt< T_SDISC.ELEMENT,
                                          N_QUANTITIES,
                                          ORDER,
                                          ORDER,
                                          N_CRUNS >::fMatId( i_vIdElFaEl[l_el][l_fa],
                                                             i_fIdElFaEl[l_el][l_fa] );

          if( (i_faChars[l_faId].spType & OUTFLOW) != OUTFLOW ) {
            // derive neighbor
            if( (i_faChars[l_faId].spType & FREE_SURFACE) != FREE_SURFACE )
              l_ne = i_elFaEl[l_el][l_fa];
            else
              l_ne = l_el;

            /*
             * prefetches
             */
            const TL_T_REAL (* l_pre)[N_ELEMENT_MODES][N_CRUNS] = nullptr;
            TL_T_INT_LID l_neUp = std::numeric_limits<TL_T_INT_LID>::max();
            // prefetch for the upcoming surface integration of this element
            if( l_fa < C_ENT[T_SDISC.ELEMENT].N_FACES-1 ) l_neUp = i_elFaEl[l_el][l_fa+1];
            // first surface integration of the next element
            else if( l_el < i_first+i_nElements-1 ) l_neUp = i_elFaEl[l_el+1][0];

            // only proceed with adjacent data if the element exists
            if( l_neUp != std::numeric_limits<TL_T_INT_LID>::max() ) l_pre = i_tInt[l_neUp];
            // next element data in case of boundary conditions
            else if( l_el < i_first+i_nElements-1 )              l_pre = io_dofs[l_el+1];
            // default to element data to avoid performance penality
            else                                                 l_pre = io_dofs[l_el];

            /*
             * solve
             */
            SurfInt< T_SDISC.ELEMENT,
                     N_QUANTITIES,
                     ORDER,
                     ORDER,
                     N_CRUNS >::neigh( ((i_faChars[l_faId].spType & FREE_SURFACE) != FREE_SURFACE ) ? i_dg.mat.fluxN[l_fId] :
                                                                                                      i_dg.mat.fluxL[l_fa],
                                       i_dg.mat.fluxT[l_fa],
                                       ( TL_T_REAL (*)[N_QUANTITIES] )  ( i_fluxSolvers[l_el][l_fa].solver[0] ), // TODO: fix struct
                                       i_tInt[l_ne],
                                       i_mm,
                                       io_dofs[l_el],
                                       l_tmpFa,
                                       l_pre,
                                       l_fa,
                                       ((i_faChars[l_faId].spType & FREE_SURFACE) != FREE_SURFACE ) ? l_fId + C_ENT[T_SDISC.ELEMENT].N_FACES: l_fa );
          }
        }
      }
    }
};

#endif
