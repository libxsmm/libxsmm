/**
 * @file This file is part of EDGE.
 *
 * @author Alexander Breuer (anbreuer AT ucsd.edu)
 *
 * @section LICENSE
 * Copyright (c) 2016-2017, Regents of the University of California
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
 * Compile time constants for the elatic wave equations.
 **/
// elastics perform three steps per time step
// 0) time prediction + local cont
// 1) neighboring cont
// 2) sources
const unsigned short N_STEPS_PER_UPDATE=3;
const unsigned short N_ENTRIES_CONTROL_FLOW=8;

const real_base C_SCALE_ENTROPY_FIX_HARTEN = 0.05;

#if PP_ORDER > 9
#error not supported
#endif
// CFL-numbers based on (experimental) 1D-stability of advection equation
const double SCALE_CFL_ORDER[9] = { 0.45, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.4};

const double SCALE_CFL=0.5*SCALE_CFL_ORDER[PP_ORDER-1];

/*
 * Bitsmasks for elastics.
 **/
static_assert( sizeof(int_spType) >= 4, "size of entity type too small" );
typedef enum: int_spType {
  FREE_SURFACE = 101,  // TODO: change to bitmask
  OUTFLOW      = 105,  // TODO: change to bitmask
  RUPTURE      = 201,  // TODO: change to bitmask
  SOURCE       = 65536 // 0b0000000000000000000000000000000000000000000000010000000000000000
} t_spTypeElastic;

#if PP_N_DIM == 2
// two normal stress components (sigma_xx, sigma_yy), one shear stress component (sigma_xy), two particle velocites (u, v)
const unsigned short N_QUANTITIES = 5;
#elif PP_N_DIM == 3
// three normal stress components (signa_xx, sigma_yy, sigmag_zz), three shear stress (sigma_xy, sigma_yz, sigma_xz), three particle velocities (u,v,w)
const unsigned short N_QUANTITIES = 9;
#else
#error elastic constants for other dimensions than 2 and 3 undefined.
#endif

/**
 * Scratch memory (per thread)
 **/
#define PP_SCRATCH_MEMORY
#define PP_SCRATCH_MEMORY_HBW
struct scratchMem {
  // temporary results
  real_base tRes[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS] __attribute__ ((aligned (ALIGNMENT.ELEMENT_MODES.PRIVATE)));
  // derivative buffer
  real_base dBuf[ORDER][N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS] __attribute__ ((aligned (ALIGNMENT.ELEMENT_MODES.PRIVATE)));
};
typedef scratchMem t_scratchMem;


/*
 * First global data encodes the setup
 */
#define PP_N_GLOBAL_PRIVATE_1 1
struct setupChars {
  enum InitialDofs { zero, planeWaves } initialDofs;
};
typedef setupChars t_globalPrivate1;

#if 0
/*
 * Source terms
 */
#include "src/impl/elastic/solvers/Kinematics.type"
// maximum number of independent sources if given by the number of fused runs
#define PP_N_GLOBAL_SHARED_3 PP_N_CRUNS
typedef edge::elastic::solvers::t_Kinematics< N_DIM,
                                              N_ELEMENT_MODES,
                                              1,
                                              real_base,
                                              int_el > t_globalShared3;

// positions of first point source associated to the sparse source elements
#define PP_N_GLOBAL_SHARED_4 1
typedef int_el (*t_globalShared4)[N_CRUNS];
#endif

#if 0
/*
 * Rupture physics
 */
#include "src/impl/elastic/solvers/FrictionLaws.type"
#define PP_N_GLOBAL_SHARED_2 1
typedef edge::elastic::solvers::t_LinSlipWeakGlobal<real_base, N_CRUNS> t_globalShared2;
#define PP_N_FACE_SPARSE_SHARED_1 1
typedef edge::elastic::solvers::t_LinSlipWeakFace<real_base> t_faceSparseShared1;
#define PP_N_FACE_SPARSE_SHARED_2 1
typedef edge::elastic::solvers::t_LinSlipWeakFaceQuadPoint<real_base, N_DIM, N_CRUNS>
          t_faceSparseShared2[CE_N_FACE_QUAD_POINTS( T_SDISC.ELEMENT, ORDER )];

// link between sparse rupture faces and sparse rupture elements
#define PP_N_FACE_SPARSE_SHARED_3 2
typedef int_el t_faceSparseShared3;
// updates for left and right side
#define PP_N_FACE_SPARSE_SHARED_4 2
typedef real_base t_faceSparseShared4[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS];
// rupture "solvers": tm1, solMsJumpL, solMsFluxL, soMsFluxR
#define PP_N_FACE_SPARSE_SHARED_5 4
typedef real_base t_faceSparseShared5[N_QUANTITIES][N_QUANTITIES];

// internal boundary data of the rupture faces
#include "src/impl/elastic/solvers/InternalBoundary.type"
#define PP_N_FACE_SPARSE_SHARED_6 1
typedef edge::elastic::solvers::t_InternalBoundaryFace< real_base, t_spTypeElastic > t_faceSparseShared6;

// link between sparse rupture elements and the sparse rupture faces
#define PP_N_ELEMENT_SPARSE_SHARED_2 1
typedef int_el t_elementSparseShared2[C_ENT[T_SDISC.ELEMENT].N_FACES];
// DOFs of the rupture elements
#define PP_N_ELEMENT_SPARSE_SHARED_3 1
typedef real_base t_elementSparseShared3[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS];
#endif

/*
 * First shared element data are the background parameters.
 */
#define PP_N_ELEMENT_SHARED_1 1
typedef struct {
  // density rho
  real_base rho;
  // lame parameter lambda
  real_base lam;
  // lame parameter mu
  real_base mu;
} t_bgPars;
typedef t_bgPars t_elementShared1;

typedef struct {
  real_base solver[N_QUANTITIES][N_QUANTITIES];
} t_fluxSolver;

/*
 * Second shared element data are the flux solvers for the element's own contribution.
 */
#define PP_N_ELEMENT_SHARED_2 C_ENT[T_SDISC.ELEMENT].N_FACES
typedef t_fluxSolver t_elementShared2;

/*
 * Third shared element data are the flux solvers for the neighboring elements' contribution.
 */
#define PP_N_ELEMENT_SHARED_3 C_ENT[T_SDISC.ELEMENT].N_FACES
typedef t_fluxSolver t_elementShared3;

// setup number of entries in the star matrices
#if defined PP_T_KERNELS_XSMM
const unsigned short N_MAT_STAR = (N_DIM==2) ? 10 : 24;
#endif

/*
 * Fourth shared element data are the star matrices.
 */
typedef struct {
#if defined PP_T_KERNELS_XSMM
  real_base mat[N_MAT_STAR];
#else
  real_base mat[N_QUANTITIES][N_QUANTITIES];
#endif
} t_matStar;

#define PP_N_ELEMENT_SHARED_4 N_DIM
typedef t_matStar t_elementShared4;

/*
 * First private element mode data are the first set of dofs.
 */
#define PP_N_ELEMENT_MODE_PRIVATE_1 N_QUANTITIES
typedef real_base t_elementModePrivate1;

/*
 * Second private element mode data are the time integrated dofs.
 */
#define PP_N_ELEMENT_MODE_PRIVATE_2 N_QUANTITIES
#define PP_ELEMENT_MODE_PRIVATE_2_HBW
typedef real_base t_elementModePrivate2;
