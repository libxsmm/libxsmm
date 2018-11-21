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

namespace edge {
  namespace reproducers {
    int full( unsigned int const i_nSteps, unsigned int const i_nElements );
  }
}

int main(int i_argc, char *i_argv[]) {
  std::cout << std::endl;
  std::cout << "EDGE AderDg Full (local+neigh) Reproducer" << std::endl;

  // Parse cmd arguments for hyper-parameters
  unsigned int l_nSteps;
  unsigned int l_nElements;
  if ( i_argc == 3 ) {
    l_nSteps    = (unsigned int)atoi(i_argv[1]);
    l_nElements = (unsigned int)atoi(i_argv[2]);
  } else {
    std::cout << "Usage: ./" << i_argv[0] << " {NUM_STEPS} {NUM_ELEMENTS} [-h|--help]\n" << std::endl;
    std::exit(1);
  }
  std::cout << "Order: " << ORDER << ", Precision: " << PP_PRECISION << ", Fused runs: " << N_CRUNS << std::endl;
  std::cout << "#Steps: " << l_nSteps << ", #Elements: " << l_nElements << std::endl;
  std::cout << std::endl;

  edge::reproducers::full( l_nSteps, l_nElements );

  return 0;
}


int edge::reproducers::full( unsigned int const i_nSteps, unsigned int const i_nElements ) {

  double                                  l_dT = 0.000001;
  unsigned int                            l_nSteps = i_nSteps;
  unsigned int                            l_nElements = i_nElements;
  t_elementChars                        * l_elChars; /* zero initialization */
  t_faceChars                           * l_faChars;
  unsigned int                         (* l_elFa)[ C_ENT[T_SDISC.ELEMENT].N_FACES ];
  unsigned int                         (* l_elFaEl)[ C_ENT[T_SDISC.ELEMENT].N_FACES ];
  unsigned short                       (* l_fIdElFaEl)[ C_ENT[T_SDISC.ELEMENT].N_FACES ];
  unsigned short                       (* l_vIdElFaEl)[ C_ENT[T_SDISC.ELEMENT].N_FACES ];
  t_dg                                    l_dg;
  t_matStar                            (* l_starM)[N_DIM];
  t_fluxSolver                         (* l_fluxSolvers)[ C_ENT[T_SDISC.ELEMENT].N_FACES ];
  real_base                            (* l_dofs)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS];
  real_base                            (* l_tInt)[N_QUANTITIES][N_ELEMENT_MODES][N_CRUNS];
  edge::io::Receivers                     l_recvs;
  edge::data::MmXsmmFused< real_base >    l_mm;
  unsigned int                            l_dummyUInt;
  double                                  l_dummyDouble;


  // 1. Set up structures
  setupDg( l_dg );
  setupKernel( l_mm );

  setupStarM( l_nElements, &l_starM );
  setupFluxSolv( l_nElements, &l_fluxSolvers );

  setupTensor( l_nElements, &l_dofs, &l_tInt );
#ifdef PP_USE_OMP
  #pragma omp parallel
  #pragma omp critical
#endif
  setupScratchMem( &(edge::parallel::g_scratchMem) );

  setupPseudoMesh( edge::reproducers::C_MODE_FULL, l_nElements,
                   &l_elChars,
                   &l_faChars, &l_elFa, &l_elFaEl, &l_fIdElFaEl, &l_vIdElFaEl );


  // 2. Run solvers
  std::cout << "Runing solvers" << std::endl;
  unsigned long long l_start = libxsmm_timer_tick();
#ifdef PP_USE_OMP
  #pragma omp parallel firstprivate( l_nSteps, l_nElements, l_dT )  \
                       firstprivate( l_elChars )                    \
                       firstprivate( l_dg, l_starM, l_fluxSolvers ) \
                       firstprivate( l_dofs, l_tInt  )              \
                       firstprivate( l_mm )                         \
                       private( l_recvs, l_dummyUInt, l_dummyDouble )
#endif
  {
    const unsigned int l_nThreads = omp_get_num_threads();
    const unsigned int l_tid = omp_get_thread_num();
    unsigned int l_firstEl = (unsigned int)((l_nElements + l_nThreads - 1) / l_nThreads) * l_tid;
    unsigned int l_lastEl = (unsigned int)((l_nElements + l_nThreads - 1) / l_nThreads) * (l_tid + 1);
    l_lastEl = std::min(l_lastEl, l_nElements);
    unsigned int l_numEl = l_lastEl - l_firstEl;

    for ( unsigned int l_step = 0; l_step < l_nSteps; l_step++ ) {
      edge::elastic::solvers::AderDg::local< unsigned int,
                                             real_base,
                                             edge::data::MmXsmmFused< real_base > >
                                           ( l_firstEl,
                                             l_numEl,
                                             l_dummyDouble,
                                             l_dT,
                                             l_dummyUInt,
                                             l_dummyUInt,
                                             nullptr,
                                             nullptr,
                                             l_elChars,
                                             l_dg,
                                             l_starM,
                                             l_fluxSolvers,
                                             l_dofs,
                                             l_tInt,
                                             nullptr,
                                             l_recvs,
                                             l_mm           );
#ifdef PP_USE_OMP
      #pragma omp barrier
#endif
      edge::elastic::solvers::AderDg::neigh< unsigned int,
                                             real_base,
                                             edge::data::MmXsmmFused< real_base > >
                                           ( l_firstEl,
                                             l_numEl,
                                             l_dummyUInt,
                                             l_dg,
                                             l_faChars,
                                             l_fluxSolvers,
                                             l_elFa,
                                             l_elFaEl,
                                             nullptr,
                                             nullptr,
                                             l_fIdElFaEl,
                                             l_vIdElFaEl,
                                             l_tInt,
                                             nullptr,
                                             l_dofs,
                                             l_mm           );
#ifdef PP_USE_OMP
      #pragma omp barrier
#endif
    }
  }
  unsigned long long l_end = libxsmm_timer_tick();

  // 3. Print statistics
  double l_time = libxsmm_timer_duration(l_start, l_end);
  unsigned int l_total_flops[] =
  {
    1584,6642,19944,52002,121032,260370,520038
  };
  unsigned long long l_flops = (unsigned long long)l_total_flops[ORDER-1] * PP_N_CRUNS * \
                               l_nElements * l_nSteps;
  double l_gflops = (double)l_flops / (l_time * 1000000000);
  std::cout << "Elapsed time: " << l_time << " s" << std::endl;
  std::cout << "Performance:  " << l_gflops << " GFLOPS" << std::endl;
  std::cout << std::endl;

#ifdef PP_REPRODUCER_DUMP
  std::string l_dumpFileName1 = "./combine_o"+std::to_string(ORDER)+"_"
                                "f"+std::to_string(PP_PRECISION)+"_"
                                "el"+std::to_string(l_nElements)+"_"
                                "stp"+std::to_string(l_nSteps)+"_dofs.log";
  std::string l_dumpFileName2 = "./combine_o"+std::to_string(ORDER)+"_"
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


  // 4. Clean up
  cleanupDg( l_dg );
  cleanupStarM( l_starM );
  cleanupFluxSolv( l_fluxSolvers );
  cleanupTensor( l_dofs, l_tInt );

#ifdef PP_USE_OMP
  #pragma omp parallel
  #pragma omp critical
#endif
  cleanupScratchMem( edge::parallel::g_scratchMem );

  cleanupPseudoMesh( edge::reproducers::C_MODE_NEIGH,
                     l_elChars,
                     l_faChars, l_elFa, l_elFaEl, l_fIdElFaEl, l_vIdElFaEl );

  return 0;
}

