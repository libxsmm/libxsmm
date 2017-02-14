/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "common_edge_proxy.h"
#include <libxsmm.h>

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf( stderr, "arguments: M #iters CSR-file!\n" );
    exit(-1);
  }

  unsigned int N_ELEMENT_MODES = atoi(argv[1]);
  unsigned int REPS = atoi(argv[2]);
  char* l_csr_file = argv[3];

  REALTYPE* l_a_de = (REALTYPE*)_mm_malloc(N_QUANTITIES * N_QUANTITIES * sizeof(REALTYPE), 64);
  REALTYPE* l_a_sp; 
  REALTYPE* l_b = (REALTYPE*)_mm_malloc(N_QUANTITIES * N_ELEMENT_MODES * N_CRUNS* sizeof(REALTYPE), 64);
  unsigned int* l_rowptr;
  unsigned int* l_colidx;
  unsigned int l_rowcount, l_colcount, l_elements;
  REALTYPE* l_c = (REALTYPE*)_mm_malloc(N_QUANTITIES * N_ELEMENT_MODES * N_CRUNS * sizeof(REALTYPE), 64);
  REALTYPE* l_c_gold = (REALTYPE*)_mm_malloc(N_QUANTITIES * N_ELEMENT_MODES * N_CRUNS * sizeof(REALTYPE), 64);
  REALTYPE* l_c_asm = (REALTYPE*)_mm_malloc(N_QUANTITIES * N_ELEMENT_MODES * N_CRUNS * sizeof(REALTYPE), 64);
  REALTYPE l_max_error = 0.0;
  unsigned int l_i;
  unsigned int l_j;
  unsigned int l_k;
  unsigned int l_jj;
  unsigned int l_n;

  REALTYPE (*l_p_b)     [N_ELEMENT_MODES][N_CRUNS] = (REALTYPE (*)[N_ELEMENT_MODES][N_CRUNS])l_b; 
  REALTYPE (*l_p_c)     [N_ELEMENT_MODES][N_CRUNS] = (REALTYPE (*)[N_ELEMENT_MODES][N_CRUNS])l_c; 
  REALTYPE (*l_p_c_asm) [N_ELEMENT_MODES][N_CRUNS] = (REALTYPE (*)[N_ELEMENT_MODES][N_CRUNS])l_c_asm; 
  REALTYPE (*l_p_c_gold)[N_ELEMENT_MODES][N_CRUNS] = (REALTYPE (*)[N_ELEMENT_MODES][N_CRUNS])l_c_gold; 

  struct timeval l_start, l_end;
  double l_total;

  /* touch B */
  for ( l_i = 0; l_i < N_QUANTITIES; l_i++) {
    for ( l_j = 0; l_j < N_ELEMENT_MODES; l_j++) {
      for ( l_k = 0; l_k < N_CRUNS; l_k++ ) {
        l_p_b[l_i][l_j][l_k] = (REALTYPE)drand48();
      }
    }
  }

  /* touch C */
  for ( l_i = 0; l_i < N_QUANTITIES; l_i++) {
    for ( l_j = 0; l_j < N_ELEMENT_MODES; l_j++) {
      for ( l_k = 0; l_k < N_CRUNS; l_k++ ) {
        l_p_c[l_i][l_j][l_k] = (REALTYPE)0.0;
        l_p_c_gold[l_i][l_j][l_k] = (REALTYPE)0.0;
        l_p_c_asm[l_i][l_j][l_k] = (REALTYPE)0.0;
      }
    }
  }

  /* read A, CSR */
  libxsmm_sparse_csr_reader(  l_csr_file,
                             &l_rowptr,
                             &l_colidx,
                             &l_a_sp,
                             &l_rowcount, &l_colcount, &l_elements );

  /* copy b to dense */
  printf("CSR matrix data structure we just read:\n");
  printf("rows: %u, columns: %u, elements: %u\n", l_rowcount, l_colcount, l_elements);

  for ( l_n = 0; l_n < (N_QUANTITIES * N_QUANTITIES); l_n++) {
    l_a_de[l_n] = 0.0;
  }

  for ( l_n = 0; l_n < N_QUANTITIES; l_n++) {
    const unsigned int l_rowelems = l_rowptr[l_n+1] - l_rowptr[l_n];
    assert(l_rowptr[l_n+1] >= l_rowptr[l_n]);

    for ( l_k = 0; l_k < l_rowelems; l_k++) {
      l_a_de[(l_n * N_QUANTITIES) + l_colidx[l_rowptr[l_n] + l_k]] = l_a_sp[l_rowptr[l_n] + l_k];
    }
  }

  /* dense routine */
  gettimeofday(&l_start, NULL);
#if 1
  for ( l_n = 0; l_n < REPS; l_n++) {
    for ( l_i = 0; l_i < N_QUANTITIES; l_i++) {
      for ( l_j = 0; l_j < N_ELEMENT_MODES; l_j++) {
        for ( l_jj = 0; l_jj < N_QUANTITIES; l_jj++) {
          #pragma simd
          for (l_k = 0; l_k < N_CRUNS; l_k++) {
            l_p_c_gold[l_i][l_j][l_k] += l_a_de[(l_i*N_QUANTITIES)+l_jj] * l_p_b[l_jj][l_j][l_k] ;
          }
        }
      }
    }
  }
#endif
  gettimeofday(&l_end, NULL);
  l_total = sec(l_start, l_end);
  printf("%fs for dense\n", l_total);
  printf("%f GFLOPS for dense\n", ((double)((double)REPS * (double)N_QUANTITIES * (double)N_QUANTITIES * (double)N_ELEMENT_MODES * (double)N_CRUNS) * 2.0) / (l_total * 1.0e9));

  /* sparse routine */
  gettimeofday(&l_start, NULL);
  for ( l_n = 0; l_n < REPS; l_n++) {
    for ( l_i = 0; l_i < N_QUANTITIES; l_i++) {
      for ( l_j = 0; l_j < N_ELEMENT_MODES; l_j++) {
        unsigned int l_elems_per_row = l_rowptr[l_i+1] - l_rowptr[l_i];
        unsigned int l_rowstart = l_rowptr[l_i];
        for ( l_jj = 0; l_jj < l_elems_per_row; l_jj++) {
          #pragma simd
          for (l_k = 0; l_k < N_CRUNS; l_k++) {
            l_p_c[l_i][l_j][l_k] += l_a_sp[l_rowstart+l_jj] * l_p_b[l_colidx[l_rowptr[l_i] + l_jj]][l_j][l_k];
          }
        }
      }
    }
  }
  gettimeofday(&l_end, NULL);
  l_total = sec(l_start, l_end);
  printf("%fs for sparse\n", l_total);
  printf("%f GFLOPS for sparse\n", ((double)((double)REPS * (double)N_QUANTITIES * (double)l_elements * (double)N_CRUNS) * 2.0) / (l_total * 1.0e9));


  /* sparse routine */
  libxsmm_gemm_descriptor l_xgemm_desc;
  LIBXSMM_GEMM_DESCRIPTOR(l_xgemm_desc, 1,
    (0 == 0 ? 0 : LIBXSMM_GEMM_FLAG_F32PREC)
      | (0 != 0 ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0)
      | (0 != 0 ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0),
    N_QUANTITIES, N_ELEMENT_MODES, N_QUANTITIES, 0, N_ELEMENT_MODES, N_ELEMENT_MODES,
    1.0, 1.0, LIBXSMM_PREFETCH_NONE);
  libxsmm_dmmfunction mykernel = libxsmm_create_dcsr_soa( &l_xgemm_desc, l_rowptr, l_colidx, l_a_sp ).dmm;

  gettimeofday(&l_start, NULL);
  for ( l_n = 0; l_n < REPS; l_n++) {
    mykernel( l_a_sp, &(l_p_b[0][0][0]), &(l_p_c_asm[0][0][0]) );
  }
  gettimeofday(&l_end, NULL);
  l_total = sec(l_start, l_end);
  printf("%fs for sparse (asm)\n", l_total);
  printf("%f GFLOPS for sparse (asm)\n", ((double)((double)REPS * (double)N_QUANTITIES * (double)l_elements * (double)N_CRUNS) * 2.0) / (l_total * 1.0e9));
  /* check for errors */
  for ( l_i = 0; l_i < N_QUANTITIES; l_i++) {
    for ( l_j = 0; l_j < N_ELEMENT_MODES; l_j++) {
      for ( l_k = 0; l_k < N_CRUNS; l_k++ ) {
        if (fabs(l_p_c_gold[l_i][l_j][l_k]-l_p_c[l_i][l_j][l_k]) > l_max_error ) {
          l_max_error = fabs(l_p_c_gold[l_i][l_j][l_k]-l_p_c[l_i][l_j][l_k]);
        }
      }
    }
  }
  printf("max error: %f\n", l_max_error);

  /* check for errors */
  l_max_error = (REALTYPE)0.0;
  for ( l_i = 0; l_i < N_QUANTITIES; l_i++) {
    for ( l_j = 0; l_j < N_ELEMENT_MODES; l_j++) {
      for ( l_k = 0; l_k < N_CRUNS; l_k++ ) {
        if (fabs(l_p_c_gold[l_i][l_j][l_k]-l_p_c_asm[l_i][l_j][l_k]) > l_max_error ) {
          l_max_error = fabs(l_p_c_gold[l_i][l_j][l_k]-l_p_c_asm[l_i][l_j][l_k]);
        }
      }
    }
  }
  printf("max error: %f\n", l_max_error);
  /* free */
}
