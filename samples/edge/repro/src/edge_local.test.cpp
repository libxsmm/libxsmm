#include <iostream>
#include <string>
#include <cstdlib>

#ifdef PP_USE_OMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num()  0
#endif

#include "parallel_global.hpp"
#include "AderDg.hpp"
#include "edge_setup.hpp"

#if defined(PP_T_KERNELS_XSMM) || defined(PP_T_KERNELS_XSMM_DENSE_SINGLE)
#include "libxsmm.h"
#endif


int main(int i_argc, char *i_argv[]) {
  std::cout << std::endl;
  std::cout << "EDGE Local Update Reproducer" << std::endl;

  double           l_dT = 0.000001;
  unsigned int     l_nSteps;
  unsigned int     l_nElements;
  real_base        l_stiffT[ N_DIM ][N_ELEMENT_MODES][N_ELEMENT_MODES];
  real_base        l_stiff [ N_DIM ][N_ELEMENT_MODES][N_ELEMENT_MODES];
  real_base        l_fluxL[ C_ENT[T_SDISC.ELEMENT].N_FACES ][N_ELEMENT_MODES][N_FACE_MODES];
  real_base        l_fluxN[ N_FLUXN_MATRICES               ][N_ELEMENT_MODES][N_FACE_MODES];
  real_base        l_fluxT[ C_ENT[T_SDISC.ELEMENT].N_FACES ][N_FACE_MODES][N_ELEMENT_MODES];
  real_base        l_starMat[N_QUANTITIES][N_QUANTITIES];
  real_base        l_fluxSolver[N_QUANTITIES][N_QUANTITIES];
  real_base     (* l_dofs)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS];
  real_base     (* l_tInt)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS];
  real_base        l_dBuf[ORDER][N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS];
  real_base        l_scratch1[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS];
  real_base        l_scratch2[N_QUANTITIES][N_FACE_MODES][N_CRUNS];
  real_base        l_scratch3[N_QUANTITIES][N_FACE_MODES][N_CRUNS];


  // 1. Parse cmd arguments for hyper-parameters
  if ( i_argc == 3 ) {
    l_nSteps    = (unsigned int)atoi(i_argv[1]);
    l_nElements = (unsigned int)atoi(i_argv[2]);
  } else {
    std::cout << "Usage: ./local_test {NUM_STEPS} {NUM_ELEMENTS} [-h|--help]\n" << std::endl;
    std::exit(1);
  }
  std::cout << "#Steps: " << l_nSteps << ", #Elements: " << l_nElements << std::endl;
  std::cout << std::endl;

  // 2. Set up structures
  edge::reproducers::setupTensor( l_nElements, &l_dofs, &l_tInt );

  std::vector< real_base > l_matDense;
  for (unsigned short l_di = 0; l_di < N_DIM; l_di++) {
    edge::reproducers::readSparseMatrixDense( C_STIFFT_NAME(l_di), l_matDense );
    for (unsigned int l_i = 0; l_i < N_ELEMENT_MODES; l_i++)
      for (unsigned int l_j = 0; l_j < N_ELEMENT_MODES; l_j++)
        l_stiffT[l_di][l_i][l_j] = l_matDense[l_i*N_ELEMENT_MODES+l_j];
  }
  for (unsigned short l_di = 0; l_di < N_DIM; l_di++) {
    edge::reproducers::readSparseMatrixDense( C_STIFF_NAME(l_di), l_matDense );
    for (unsigned int l_i = 0; l_i < N_ELEMENT_MODES; l_i++)
      for (unsigned int l_j = 0; l_j < N_ELEMENT_MODES; l_j++)
        l_stiff[l_di][l_i][l_j] = l_matDense[l_i*N_ELEMENT_MODES+l_j];
  }
  for (unsigned short l_fl = 0; l_fl < C_ENT[T_SDISC.ELEMENT].N_FACES; l_fl++) {
    edge::reproducers::readSparseMatrixDense( C_FLUXL_NAME(l_fl), l_matDense );
    for (unsigned int l_i = 0; l_i < N_ELEMENT_MODES; l_i++)
      for (unsigned int l_j = 0; l_j < N_FACE_MODES; l_j++)
        l_fluxL[l_fl][l_i][l_j] = l_matDense[l_i*N_FACE_MODES+l_j];
  }
  for (unsigned short l_fn = 0; l_fn < N_FLUXN_MATRICES; l_fn++) {
    edge::reproducers::readSparseMatrixDense( C_FLUXN_NAME(l_fn), l_matDense );
    for (unsigned int l_i = 0; l_i < N_ELEMENT_MODES; l_i++)
      for (unsigned int l_j = 0; l_j < N_FACE_MODES; l_j++)
        l_fluxN[l_fn][l_i][l_j] = l_matDense[l_i*N_FACE_MODES+l_j];
  }
  for (unsigned short l_ft = 0; l_ft < C_ENT[T_SDISC.ELEMENT].N_FACES; l_ft++) {
    edge::reproducers::readSparseMatrixDense( C_FLUXT_NAME(l_ft), l_matDense );
    for (unsigned int l_i = 0; l_i < N_FACE_MODES; l_i++)
      for (unsigned int l_j = 0; l_j < N_ELEMENT_MODES; l_j++)
        l_fluxT[l_ft][l_i][l_j] = l_matDense[l_i*N_ELEMENT_MODES+l_j];
  }
  edge::reproducers::readSparseMatrixDense( C_MSTAR_NAME(), l_matDense );
  for (unsigned int l_i = 0; l_i < N_QUANTITIES; l_i++)
    for (unsigned int l_j = 0; l_j < N_QUANTITIES; l_j++)
      l_starMat[l_i][l_j] = l_matDense[l_i*N_QUANTITIES+l_j];
  edge::reproducers::readSparseMatrixDense( C_FLUXSOLV_NAME(), l_matDense );
  for (unsigned int l_i = 0; l_i < N_QUANTITIES; l_i++)
    for (unsigned int l_j = 0; l_j < N_QUANTITIES; l_j++)
      l_fluxSolver[l_i][l_j] = l_matDense[l_i*N_QUANTITIES+l_j];


  // 3. Run dense routine
  for ( unsigned int l_step = 0; l_step < l_nSteps; l_step++ ) {
    for ( unsigned int l_el = 0; l_el < l_nElements; l_el++ ) {
      // 1) Time Evaluation
      real_base l_scalar = l_dT;
      for ( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
        for ( unsigned short l_md = 0; l_md < N_ELEMENT_MODES; l_md++ )
          for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ ) {
            l_dBuf[0][l_qt][l_md][l_cfr] = l_dofs[l_el][l_qt][l_md][l_cfr];
            l_tInt[l_el][l_qt][l_md][l_cfr] = l_scalar * l_dofs[l_el][l_qt][l_md][l_cfr];
          }

      for ( unsigned short l_de = 1; l_de < ORDER; l_de++ ) {
        for ( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
          for ( unsigned short l_md = 0; l_md < N_ELEMENT_MODES; l_md++ )
            for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ ) l_dBuf[l_de][l_qt][l_md][l_cfr] = 0;

        for ( unsigned short l_di = 0; l_di < N_DIM; l_di++ ) {
          for ( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
            for ( unsigned short l_md = 0; l_md < N_ELEMENT_MODES; l_md++ ) {
              for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ ) l_scratch1[l_qt][l_md][l_cfr] = 0;
              for ( unsigned short l_k_md = 0; l_k_md < N_ELEMENT_MODES; l_k_md++ )
                for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ )
                  l_scratch1[l_qt][l_md][l_cfr] += l_dBuf[l_de-1][l_qt][l_k_md][l_cfr]*l_stiffT[l_di][l_k_md][l_md];
            }

          for ( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
            for ( unsigned short l_md = 0; l_md < N_ELEMENT_MODES; l_md++ )
              for ( unsigned short l_k_qt = 0; l_k_qt < N_QUANTITIES; l_k_qt++ )
                for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ )
                  l_dBuf[l_de][l_qt][l_md][l_cfr] += l_starMat[l_qt][l_k_qt] * l_scratch1[l_k_qt][l_md][l_cfr];
        } // 3 dims for time derivatives

        l_scalar *= -l_dT / (l_de+1);

        for ( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
          for ( unsigned short l_md = 0; l_md < N_ELEMENT_MODES; l_md++ )
            for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ )
              l_tInt[l_el][l_qt][l_md][l_cfr] += l_scalar * l_dBuf[l_de][l_qt][l_md][l_cfr];
      } // all orders for Time CK

      // 2) Volume Integration
      for ( unsigned short l_di = 0; l_di < N_DIM; l_di++ ) {
        for ( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
          for ( unsigned short l_md = 0; l_md < N_ELEMENT_MODES; l_md++ ) {
            for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ ) l_scratch1[l_qt][l_md][l_cfr] = 0;
            for ( unsigned short l_k_qt = 0; l_k_qt < N_QUANTITIES; l_k_qt++ )
              for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ )
                l_scratch1[l_qt][l_md][l_cfr] += l_starMat[l_qt][l_k_qt] * l_tInt[l_el][l_k_qt][l_md][l_cfr];
          }

        for ( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
          for ( unsigned short l_md = 0; l_md < N_ELEMENT_MODES; l_md++ )
            for ( unsigned short l_k_md = 0; l_k_md < N_ELEMENT_MODES; l_k_md++ )
              for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ )
                l_dofs[l_el][l_qt][l_md][l_cfr] += l_scratch1[l_qt][l_k_md][l_cfr]*l_stiff[l_di][l_k_md][l_md];
      } // 3 dims for VolInt

      // 3) Surface Integration (local)
      for ( unsigned short l_fa = 0; l_fa < C_ENT[T_SDISC.ELEMENT].N_FACES; l_fa++ ) {
        for ( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
          for ( unsigned short l_md = 0; l_md < N_FACE_MODES; l_md++ ) {
            for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ ) l_scratch2[l_qt][l_md][l_cfr] = 0;
            for ( unsigned short l_k_md = 0; l_k_md < N_ELEMENT_MODES; l_k_md++ )
              for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ )
                l_scratch2[l_qt][l_md][l_cfr] += l_tInt[l_el][l_qt][l_k_md][l_cfr]*l_fluxL[l_fa][l_k_md][l_md];
          }

        for ( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
          for ( unsigned short l_md = 0; l_md < N_FACE_MODES; l_md++ ) {
            for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ ) l_scratch3[l_qt][l_md][l_cfr] = 0;
            for ( unsigned short l_k_qt = 0; l_k_qt < N_QUANTITIES; l_k_qt++ )
              for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ )
                l_scratch3[l_qt][l_md][l_cfr] += l_fluxSolver[l_qt][l_k_qt] * l_scratch2[l_k_qt][l_md][l_cfr];
          }

        for ( unsigned short l_qt = 0; l_qt < N_QUANTITIES; l_qt++ )
          for ( unsigned short l_md = 0; l_md < N_ELEMENT_MODES; l_md++ )
            for ( unsigned short l_k_md = 0; l_k_md < N_FACE_MODES; l_k_md++ )
              for ( unsigned short l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ )
                l_dofs[l_el][l_qt][l_md][l_cfr] += l_scratch3[l_qt][l_k_md][l_cfr]*l_fluxT[l_fa][l_k_md][l_md];
      } // 4 faces for SurfInt
    } // loop over elements
  } // loop over steps


  // 4. Dump results

#ifdef PP_REPRODUCER_DUMP
  std::string l_dumpFileName1 = "./dense_local_o"+std::to_string(ORDER)+"_"
                                "f"+std::to_string(PP_PRECISION)+"_"
                                "el"+std::to_string(l_nElements)+"_"
                                "stp"+std::to_string(l_nSteps)+"_dofs.log";
  std::string l_dumpFileName2 = "./dense_local_o"+std::to_string(ORDER)+"_"
                                "f"+std::to_string(PP_PRECISION)+"_"
                                "el"+std::to_string(l_nElements)+"_"
                                "stp"+std::to_string(l_nSteps)+"_tInt.log";
  std::ofstream l_fp1( l_dumpFileName1 );
  std::ofstream l_fp2( l_dumpFileName2 );
  for ( unsigned int l_el = 0; l_el < l_nElements; l_el++ ) {
    for ( unsigned int l_qt = 0; l_qt < N_QUANTITIES; l_qt++ ) {
      for ( unsigned int l_md = 0; l_md < N_ELEMENT_MODES; l_md++ ) {
        for ( unsigned int l_cfr = 0; l_cfr < N_CRUNS; l_cfr++ ) {
          l_fp1 << l_dofs[l_el][l_qt][l_md][l_cfr] << "\n";
          l_fp2 << l_tInt[l_el][l_qt][l_md][l_cfr] << "\n";
        }
      }
    }
  }
#endif


  // 5. Clean up
  free(l_dofs);
  free(l_tInt);


  return 0;
}
