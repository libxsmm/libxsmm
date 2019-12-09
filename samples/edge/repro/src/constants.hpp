/**
 * @file This file is part of EDGE.
 *
 * @author Alexander Breuer (anbreuer AT ucsd.edu)
 *
 * @section LICENSE
 * Copyright (c) 2015-2017, Regents of the University of California
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
 * Definition of compile time constants.
 **/
#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// entity types
typedef enum {
  POINT    = 0,
  LINE     = 1,
  QUAD4R   = 2,
  TRIA3    = 3,
  HEX8R    = 4,
  TET4     = 5
} t_entityType;

// characteristics of entities
constexpr struct {
  const unsigned short N_DIM;
  const unsigned short N_VERTICES;
  const unsigned short N_FACES;
  const unsigned short N_FACE_VERTICES;
  const t_entityType   TYPE_FACES;
}
C_ENT[6] = {
  // POINT
  { 0, 1, 1, 1, (t_entityType) -1 },
  // LINE
  { 1, 2, 2, 1, POINT             },
  // QUAD4R
  { 2, 4, 4, 2, LINE              },
  // TRIA
  { 2, 3, 3, 2, LINE              },
  // HEX8R
  { 3, 8, 6, 4, QUAD4R            },
  // TET4
  { 3, 4, 4, 3, TRIA3             }
};

/**
 * Determines the maximum of the two input values.
 *
 * @param i_first first value.
 * @param i_second second value.
 * @retrun maximum of both values.
 **/
template< typename T>
constexpr T CE_MAX( T i_first, T i_second ) {
  return i_first > i_second ? i_first : i_second;
}

/**
 * Determines the maximum multiple of two, which divides the number.
 *
 * @param i_number number which multiples are determined.
 * @param i_div divisor in this step (required for recursion, leave empty for initial call).
 * @return greatest multiple of two dividing the number.
 **/
template< typename T>
constexpr unsigned short CE_MUL_2( T i_number, unsigned short i_div = 2 ) {
  return (i_number % i_div != 0 || i_number == 0) ? i_div/2 : CE_MUL_2( i_number, i_div*2 );
}

/**
 * Returns the number of vertex options (orientations) of two neighboring faces.
 *
 * @param i_enType entity type.
 * @return number of options
 */
constexpr unsigned short CE_N_FACE_VERTEX_OPTS( t_entityType i_enType ) {
  return (i_enType == TET4) ? 3 : (
           (C_ENT[i_enType].N_DIM != 3 || i_enType == HEX8R ) ? 1 :  (unsigned short) -1
         );
}

/**
 * Gets the number of modes for the given order.
 *
 * @param i_entType entity type.
 * @param i_order order of the method.
 * @return number of modes.
 **/
constexpr unsigned short CE_N_ELEMENT_MODES( t_entityType   i_enType,
                                             unsigned short i_order ) {
  return  (i_enType == POINT) ? 1 : (
           (i_enType == LINE) ? i_order : (
             (i_enType == QUAD4R) ? i_order * i_order : (
               (i_enType == TRIA3) ? ( i_order * i_order + i_order ) / 2 : (
                 (i_enType == HEX8R) ? i_order * i_order * i_order : (
                   (i_enType == TET4) ? (i_order * (i_order+1) * (i_order+2) ) / 6 : (unsigned short) -1
                 )
               )
             )
           )
         );
}

/**
 * Gets the number of active element modes in the recursive Cauchy-Kowalevski procedure.
 *
 * @param i_elType element type.
 * @param i_order order of the CK-procedure.
 * @param i_der current time derivative (first equals to 1).
 **/
constexpr unsigned int CE_N_ELEMENT_MODES_CK( t_entityType   i_elType,
                                              unsigned short i_order,
                                              unsigned short i_der ) {
  return (i_elType == TRIA3 || i_elType == TET4 ) ? CE_N_ELEMENT_MODES( i_elType, // hierarchical basis
                                                                        i_order-i_der) :
                                                    CE_N_ELEMENT_MODES( i_elType,
                                                                        i_order );
}

/**
 * Gets the number of quad points at the faces for the given element type and order.
 *
 * Remark: Quadrature points through collapsed coordinates are assumed.
 *         This is not the optimal number of quad points for tets.
 *
 * @param i_entType entity type.
 * @param i_order order of the method.
 * @return number of quad points.
 **/
constexpr unsigned int CE_N_FACE_QUAD_POINTS( t_entityType   i_enType,
                                              unsigned short i_order ) {
  return (i_enType == TET4) ? CE_N_ELEMENT_MODES( QUAD4R, i_order ) : // collapsed
                              CE_N_ELEMENT_MODES( C_ENT[i_enType].TYPE_FACES, i_order );
}

/**
 * Derives the number of neighboring flux matrices for a given entity type.
 *
 * @param i_enType entity type.
 * @return number of flux matrices.
 **/
constexpr unsigned short CE_N_FLUXN_MATRICES( t_entityType i_enType ) {
  return (C_ENT[i_enType].N_DIM < 3) ? C_ENT[i_enType].N_FACES : (
           (i_enType == HEX8R) ? 6 : (
             (i_enType == TET4) ? 12 : (unsigned short) -1
           )
         );
}

/**
 * Overwrite clang macro for non-supporting compilers.
 * We still want those features though..
 **/
#ifndef __has_builtin
  #define __has_builtin(x) 1
#endif

/**
 * Check for a sufficient gcc-version
 **/
#if !(defined __INTEL_COMPILER) && !(defined __clang__) && defined __GNUC__
#if __GNUC__ < 5 && __GNUC_MINOR__ < 9
#error "gcc >= 4.9 required due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=53017"
#endif
#endif

/**
 * pre-processor variables
 *
 * All data related structures, such as PP_N_ELEMENT_MODE_PRIVATE_* allow for multiple representations to allow for SoA-implementations.
 * This structures have an asterisk (*) below. The keyword "mode" refers to both, modal and nodal schemes.
 * For example an old-new-FV scheme would use PP_N_ELEMENT_MODE_PRIVATE_1 and PP_N_ELEMENT_MODE_PRIVATE_2.
 * Each of these data structures is also allowed to have an independent type, this could be a struct or native floating point precision.
 * such as t_elementModePrivate1, t_elementModePrivate2.
 *
 * --- Input: Simulation related definitions ---
 * PP_PRECISION:                Floating point precision in bits.
 * PP_N_CRUNS:                  Number of concurrent forward runs executed in a single execution of EDGE.
 * PP_ORDER                     Order of convergence.
 *
 * --- Global definitions: Independent of the mesh.
 * PP_N_DIM                       Number of dimensions used in the simulation.
 * PP_N_GLOBAL_SHARED_*:          Number of global shared data-points.
 * PP_N_GLOBAL_PRIVATE_*:         Number of global data-point per cocurrent run.
 *
 * --- Element related definitions ---
 * PP_N_ELEMENT_PRIVATE_*:        Number of private data-points stored for every element and every concurrent run.
 * PP_N_ELEMENT_SHARED_*:         Number of shared data-points stored for every element. The data is shared among concurrent runs.
 *
 * PP_N_ELEMENT_MODE_PRIVATE_*:   Number of private data-points stored for every element mode and every concurrent run.
 * PP_N_ELEMENT_MODE_SHARED_*:    Number of shared data-points stored for every element mode. The data is shared among concurrent runs.
 *
 * PP_N_ELEMENT_SPARSE_PRIVATE_*: Number of private data-points stored for every sparse element.
 * PP_N_ELEMENT_SPARSE_SHARED_*:  Number of shared data-points stored for every sparse element.
 *
 *
 * --- Face related definitions ---
 * PP_N_FACE_PRIVATE_*:           Number of private data-points stored for every face and every concurrent run.
 * PP_N_FACE_SHARED_*:            Number of shared data-points stored for every face. The data is shared among concurrent runs.
 * PP_N_FACE_MODE_PRIVATE_*:      Number of private data-points stored for every face mode and every concurrent run.
 * PP_N_FACE_MODE_SHARED_*:       Number of shared data-points stored for every face mode. The data is shared among concurrent runs.
 *
 * PP_N_FACE_SPARSE_PRIVATE_*:    Number of private data-points stored for every sparse face.
 * PP_N_FACE_SPARSE_SHARED_*:     Number of shared data-points storeed for every sparse face
 *
 * **************************************************************************************************************************************
 *
 * Derived constants (processed at compile time):
 *
 * --- Derived: Simulation related definitions.
 * N_QUANTITIES              Number of quantities.
 *
 * N_STEPS_PER_UPDATE Number of steps per DOF-update.
 * N_ENTRIES_CONTROL_FLOW Number of entries in the control flow of the simulation.
 *
 * --- Element related definitions ---
 * N_ELEMENT_MODES:          Number of modes per element.
 *
 * --- Face related definitions ---
 * N_FACE_MODES              Number of modes per face.
 *
 **/
// devive dimensions from element types
#if defined PP_T_ELEMENTS_LINE
#define PP_N_DIM 1
#elif defined PP_T_ELEMENTS_QUAD4R || defined PP_T_ELEMENTS_TRIA3
#define PP_N_DIM 2
#elif defined PP_T_ELEMENTS_HEX8R || defined PP_T_ELEMENTS_TET4
#define PP_N_DIM 3
#endif

constexpr char EDGE_LOGO[] =
"##########################################################################\n"
"##############   ##############            ###############  ##############\n"
"##############   ###############         ################   ##############\n"
"#####            #####       #####      ######                       #####\n"
"#####            #####        #####    #####                         #####\n"
"#############    #####         #####  #####                  #############\n"
"#############    #####         #####  #####      #########   #############\n"
"#####            #####         #####  #####      #########           #####\n"
"#####            #####        #####    #####        ######           #####\n"
"#####            #####       #####      #####       #####            #####\n"
"###############  ###############         ###############   ###############\n"
"###############  ##############           #############    ###############\n"
"##########################################################################\n";

// copy over variables of the preprocessor
const unsigned short N_CRUNS = PP_N_CRUNS;
const unsigned short ORDER = PP_ORDER;
const unsigned short N_DIM = PP_N_DIM;

// zero tolerances for different operations
const struct {
  double MESH;
  double LINALG;
  double BASIS;
  double SOLVER;
  double TIME; }
TOL = {
  0.000000001,
  0.000000001,
  0.0000001,
  0.0000001,
  0.0000001
};

// spatial discretizations based on entities
constexpr struct {
  const t_entityType ELEMENT;
  const t_entityType FACE;
  const t_entityType VERTEX;
} T_SDISC =
#if defined PP_T_ELEMENTS_LINE
  {LINE,     POINT,    POINT};
#elif defined PP_T_ELEMENTS_QUAD4R
  {QUAD4R,   LINE,     POINT};
#elif defined PP_T_ELEMENTS_TRIA3
  {TRIA3,    LINE,     POINT};
#elif defined PP_T_ELEMENTS_HEX8R
  {HEX8R,    QUAD4R,   POINT};
#elif defined PP_T_ELEMENTS_TET4
  {TET4,     TRIA3,    POINT };
#endif

// define basic data types
#if PP_PRECISION==32
typedef float real_base;
#elif PP_PRECISION==64
typedef double real_base;
#endif
// precision of mesh associated data (vertices, ..)
typedef double real_mesh;

// size for interger represenattion of time groups
typedef unsigned short int_tg;

// size of integer representation of per-element quantities
typedef unsigned short int_qt;

// size of integer represenation of per-element modes
typedef unsigned int int_md;

// size for integer representation of the elements/faces/vertices.
typedef int int_el;

// size of global entity ids
typedef int int_gid;

// size for interger representation in time stepping
typedef unsigned long int_ts;

// integer representation of concurrent forward runs
typedef unsigned short int_cfr;

/*
 * integer representation of sparse types.
 *
 * Bits 0-14  are reserved for mesh-input, e.g. boundary conditions.
 * Bit  15    is reserved for the receiver-flag.
 * Bits 16-31 are for application purposes.
 * Bits 32-63 are reserved for the time stepping
 */
typedef long long int_spType;

// define shared enums (of all implementations) for types
typedef enum: int_spType {
  MESH_TYPE_NONE = 16384, // 0b0000000000000000000000000000000000000000000000000100000000000000
  RECEIVER       = 32768, // 0b0000000000000000000000000000000000000000000000001000000000000000
} t_enTypeShared;

// vertex characteristics
typedef struct {
  // coordinates of the vertex
  real_mesh coords[3];

  // vertex types
  int_spType spType;
} t_vertexChars;

// face characteristics
typedef struct {
  // area
  real_mesh area;
  /*
   * outer pointing normal, origin (0,0,0), length 1.0
   * the normal always points from element 0 (left) to element 1 (right)
   *     L   |   R
   *     0   |   1
   *         |_____\
   *         |  n  /
   *         |
   *         |
   **/
  real_mesh outNormal[3];

  /*
   * face-tangents, origin (0,0,0), length.
   * the tangents are orthogonal to each other and orthogonal to the normal.
   * by convention tangent 0 points in the direction of the face's first vertex v0 towards the
   * face's second vertex v1.
   */
  real_mesh tangent0[3];
  real_mesh tangent1[3];

  /*
   * sparse type of the face.
   * in the local consistency check, we assume that MESH_TYPE_NONE refers to non-boundary faces.
   */
  int_spType spType;
} t_faceChars;

// element characteristics
typedef struct {
  // volume
  real_mesh volume;

  // insphere diameter
  real_mesh inDia;

  /*
   * sparse type of the element.
   */
  int_spType spType;
} t_elementChars;

// define connectivity information
typedef struct {
  // vertices adjacent to faces
  int_el (*faVe)[ C_ENT[T_SDISC.FACE].N_VERTICES ];

  // vertices adjacent to elements
  int_el (*elVe)[ C_ENT[T_SDISC.ELEMENT].N_VERTICES ];

  // elements adjacent to faces
  int_el (*faEl)[2];

  // faces adjacent to the elements
  int_el (*elFa)[ C_ENT[T_SDISC.ELEMENT].N_FACES ];

  // elements connected to element through faces
  int_el (*elFaEl)[ C_ENT[T_SDISC.ELEMENT].N_FACES ];

  // local face id (ref element) of the face-neighboring elements
  unsigned short (*fIdElFaEl)[ C_ENT[T_SDISC.ELEMENT].N_FACES ];

  // local vertex id (ref element) of the face-neighboring elements, matching vertex 0 of the given element.
  unsigned short (*vIdElFaEl)[ C_ENT[T_SDISC.ELEMENT].N_FACES ];
} t_connect;

/*
 * Reference elements
 *
 **** Line:
 *           0          1
 *           ------------
 *       (0,0,0)   0    (1,0,0)
 *
 *
 **** Quad:
 *
 *  (0,1,0)                (1,1,0)
 *         3  ___________ 2
 *           |     2     |
 *           |           |
 *           |3         1|
 *           |           |
 *           |_____0_____|
 *          0              1
 *   (0,0,0)                (1,0,0)
 *
 *
 **** Triangle:
 *
 *   (0,1,0)
 *           2
 *           |\
 *           | \
 *           |  \
 *           |   \
 *           |2  1\
 *           |     \
 *           |      \
 *           |___0___\
 *          0         1
 *   (0,0,0)           (1,0,0)
 *
 **** Hex:
 *   face 0: 0-3-2-1
 *   face 1: 0-1-5-4
 *   face 2: 1-2-6-5
 *   face 3: 3-7-6-2
 *   face 4: 0-4-7-3
 *   face 5: 4-5-6-7
 * we are forcing counter-clockwise storage of face vertices w.r.t. to opposite face.
 *
 *
 *           7 x*******************x 6
 *            **                  **
 *           * *                 * *
 *          *  *                *  *
 *         *   *               *   *
 *        *    *              *    *
 *     4 x*******************x 5   *
 *       *     *             *     *
 *       *   3 x************ * ****x 2
 *       *    *              *    *
 *       *   *               *   *
 *       |  /                *  *
 *  zeta | / eta             * *
 *       |/                  **
 *       x---****************x
 *     0   xi                 1
 *
 **** Tet:
 * Looking in the direction of the face's outer pointing normals,
 * we are enforcing a counter-clockwise storage w.r.t. to vertex 0 (face 0-2) / vertex 1 (face 3)
 *
 *   face 0: 0-2-1
 *   face 1: 0-1-3
 *   face 2: 0-3-2
 *   face 3: 1-2-3
 *                 zeta
 *                  3: x
 *                   *
 *                  *    *
 *                 *   *
 *                *         *
 *               *
 *              *       *      *
 *             *
 *            *                    *
 *           * origin(0): x
 *          *           *     *       *
 *         *        *              *
 *        *      *                       x
 *       *    *               *          2: eta
 *      *  *     *
 * 1: x
 * xi
 */
constexpr struct {
  // vertices of reference elements
  struct {
    const real_mesh  POINT[3][1];
    const real_mesh  LINE[3][2];
    const real_mesh  QUAD[3][4];
    const real_mesh  TRIA[3][3];
    const real_mesh  HEX[3][8];
    const real_mesh  TET[3][4];
    const real_mesh *ENT[6]; // access to verts over enums
  } VE;
  // vertices of the reference elements' faces
  // Remark: We use counter-clockwise storage, which is exploited throughout the execution.
  struct {
    const real_mesh  LINE[3][2][1];
    const real_mesh  QUAD[3][4][2];
    const real_mesh  TRIA[3][3][2];
    const real_mesh  HEX[3][6][4];
    const real_mesh  TET[3][4][3];
    const real_mesh *ENT[6]; // access to faces over enums
  } FA;
  // ids of the faces' vertices in the element ensuring a counter-clockwise storage.
  struct {
    const unsigned short  LINE[1][2];
    const unsigned short  QUAD[4][2];
    const unsigned short  TRIA[3][2];
    const unsigned short  HEX[6][4];
    const unsigned short  TET[4][3];
    const unsigned short *ENT[6]; // access to ids over enums
  } FA_VE_CC;
  // volume of the reference element
  struct {
    const real_mesh LINE;
    const real_mesh QUAD;
    const real_mesh TRIA;
    const real_mesh HEX;
    const real_mesh TET;
    const real_mesh *ENT[6]; // access to ids over enums
  } VOL;
}
C_REF_ELEMENT = {
  {
    { {0},               {0},               {0}               }, // point
    { {0,1},             {0,0},             {0,0}             }, // line
    { {0,1,1,0},         {0,0,1,1},         {0,0,0,0}         }, // quad
    { {0,1,0},           {0,0,1},           {0,0,0}           }, // tria
    { {0,1,1,0,0,1,1,0}, {0,0,1,1,0,0,1,1}, {0,0,0,0,1,1,1,1} }, // hex
    { {0,1,0,0},         {0,0,1,0},         {0,0,0,1}         }, // tet
    { C_REF_ELEMENT.VE.POINT[0],
      C_REF_ELEMENT.VE.LINE[0],
      C_REF_ELEMENT.VE.QUAD[0],
      C_REF_ELEMENT.VE.TRIA[0],
      C_REF_ELEMENT.VE.HEX[0],
      C_REF_ELEMENT.VE.TET[0] }
  },
  {
    { { {0},   {1} },
      { {0},   {0} },
      { {0},   {0} }
    },
    {
      { {0,1}, {1,1}, {1,0}, {0,0} },
      { {0,0}, {0,1}, {1,1}, {1,0} },
      { {0,0}, {0,0}, {0,0}, {0,0} }
    },
    {
      { {0,1}, {1,0}, {0,0} },
      { {0,0}, {0,1}, {1,0} },
      { {0,0}, {0,0}, {0,0} },
    },
    {
      // 0: 0-3-2-1  | 1: 0-1-5-4 | 2: 1-2-6-5 | 3: 3-7-6-2 | 4: 0-4-7-3 | 5: 4-5-6-7
      { {0,0,1,1},    {0,1,1,0},   {1,1,1,1},   {0,0,1,1},   {0,0,0,0},   {0,1,1,0} },
      { {0,1,1,0},    {0,0,0,0},   {0,1,1,0},   {1,1,1,1},   {0,0,1,1},   {0,0,1,1} },
      { {0,0,0,0},    {0,0,1,1},   {0,0,1,1},   {0,1,1,0},   {0,1,1,0},   {1,1,1,1} }
    },
    {
      // 0: 0-2-1 | 1: 0-1-3 | 2: 0-3-2 |  3: 1-2-3
      { {0,0,1},   {0,1,0},   {0,0,0},     {1,0,0} },
      { {0,1,0},   {0,0,0},   {0,0,1},     {0,1,0} },
      { {0,0,0},   {0,0,1},   {0,1,0},     {0,0,1} }
    },
    { nullptr,
      C_REF_ELEMENT.FA.LINE[0][0],
      C_REF_ELEMENT.FA.QUAD[0][0],
      C_REF_ELEMENT.FA.TRIA[0][0],
      C_REF_ELEMENT.FA.HEX[0][0],
      C_REF_ELEMENT.FA.TET[0][0]
    }
  },
  {
    { {0, 1} },                                                                             // line
    { {0, 1}, {1, 2}, {2, 3}, {3, 0} },                                                     // quad
    { {0, 1}, {1, 2}, {2, 0}        },                                                      // tria
    { {0, 3, 2, 1}, {0, 1, 5, 4}, {1, 2, 6, 5}, {3, 7, 6, 2}, {0, 4, 7, 3}, {4, 5, 6, 7} }, // hex
    { {0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3} },                                         // tet
    { nullptr,
      C_REF_ELEMENT.FA_VE_CC.LINE[0],
      C_REF_ELEMENT.FA_VE_CC.QUAD[0],
      C_REF_ELEMENT.FA_VE_CC.TRIA[0],
      C_REF_ELEMENT.FA_VE_CC.HEX[0],
      C_REF_ELEMENT.FA_VE_CC.TET[0],
    }
  },
  { 1.0, 1.0, 1.0/2.0, 1.0, 1.0/6.0,
    { nullptr,
      &C_REF_ELEMENT.VOL.LINE,
      &C_REF_ELEMENT.VOL.QUAD,
      &C_REF_ELEMENT.VOL.TRIA,
      &C_REF_ELEMENT.VOL.HEX,
      &C_REF_ELEMENT.VOL.TET
    }
  }
};

/*
 * ADER-DG discretization.
 */
const unsigned int N_ELEMENT_MODES = CE_N_ELEMENT_MODES( T_SDISC.ELEMENT, ORDER );
#if 0
#include "dg/const_ader.inc"
#endif
#include "const_ader.inc"

/*
 * Memory alignment
 */
static_assert( PP_PRECISION == 64 || PP_PRECISION ==32, "precision not supported" );
constexpr struct {
  struct {
    int STACK; // stack base pointers
    int HEAP;  // heap base points
  } BASE;
  int CRUNS;   // concurrent forward runs
  struct {
    int PRIVATE; // modes on run-private data (wrapping cruns)
    int SHARED;  // modes on shared data (no cruns)
  } ELEMENT_MODES;
}
ALIGNMENT = {
  {
    CE_MAX( 2048, CE_MUL_2(N_CRUNS*N_ELEMENT_MODES) * (PP_PRECISION==64 ? 8 : 4) ),
    CE_MAX( 4096, CE_MUL_2(N_CRUNS*N_ELEMENT_MODES) * (PP_PRECISION==64 ? 8 : 4) )
  },
  CE_MUL_2(N_CRUNS                  ) * (PP_PRECISION==64 ? 8 : 4),
  {
    CE_MUL_2(N_CRUNS*N_ELEMENT_MODES) * (PP_PRECISION==64 ? 8 : 4),
    CE_MUL_2(        N_ELEMENT_MODES) * (PP_PRECISION==64 ? 8 : 4)
  }
};
static_assert( ALIGNMENT.BASE.HEAP  >= ALIGNMENT.BASE.STACK,
               "heap alignment smaller than stack alignment" );
static_assert( ALIGNMENT.BASE.STACK >= ALIGNMENT.ELEMENT_MODES.PRIVATE,
              "stack alignment smaller than private alignment" );
static_assert( ALIGNMENT.ELEMENT_MODES.PRIVATE >= ALIGNMENT.CRUNS,
              "crun alingnment smaller than private alignemnt" );
static_assert( ALIGNMENT.ELEMENT_MODES.PRIVATE >= ALIGNMENT.ELEMENT_MODES.SHARED,
               "private alignment smaller than shared alignment" );

/*
 * Setups for the different equations and discretizations.
 */
#ifdef PP_T_EQUATIONS_ADVECTION
#include "impl/advection/const.inc"
#endif

#ifdef PP_T_EQUATIONS_ELASTIC
#if 0
#include "impl/elastic/const.inc"
#endif
#include "const_elastic.inc"
#endif

#ifdef PP_T_EQUATIONS_SWE
#include "impl/swe/const.inc"
#endif

#endif
