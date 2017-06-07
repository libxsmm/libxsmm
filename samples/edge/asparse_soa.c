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

#include <libxsmm.h>

#include <common_edge_proxy.h>

int main(int argc, char* argv[]) {
  unsigned int N_ELEMENT_MODES = ( argc == 4 ) ? atoi(argv[1]) : 20;
  unsigned int REPS = ( argc == 4 ) ? atoi(argv[2]) : 1;
  char* l_csr_file = ( argc == 4 ) ? argv[3] : "file.csr" ;

  REALTYPE* l_a_de = (REALTYPE*)libxsmm_aligned_malloc(N_QUANTITIES * N_QUANTITIES * sizeof(REALTYPE), 64);
  REALTYPE* l_a_sp;
  REALTYPE* l_b = (REALTYPE*)libxsmm_aligned_malloc(N_QUANTITIES * N_ELEMENT_MODES * N_CRUNS* sizeof(REALTYPE), 64);
  unsigned int* l_rowptr;
  unsigned int* l_colidx;
  unsigned int l_rowcount, l_colcount, l_elements;
  REALTYPE* l_c = (REALTYPE*)libxsmm_aligned_malloc(N_QUANTITIES * N_ELEMENT_MODES * N_CRUNS * sizeof(REALTYPE), 64);
  REALTYPE* l_c_gold = (REALTYPE*)libxsmm_aligned_malloc(N_QUANTITIES * N_ELEMENT_MODES * N_CRUNS * sizeof(REALTYPE), 64);
  REALTYPE* l_c_asm = (REALTYPE*)libxsmm_aligned_malloc(N_QUANTITIES * N_ELEMENT_MODES * N_CRUNS * sizeof(REALTYPE), 64);
  REALTYPE l_max_error = 0.0;
  unsigned int l_i;
  unsigned int l_j;
  unsigned int l_k;
  unsigned int l_jj;
  unsigned int l_n;

  LIBXSMM_VLA_DECL(3, REALTYPE, l_p_b, l_b, N_ELEMENT_MODES, N_CRUNS);
  LIBXSMM_VLA_DECL(3, REALTYPE, l_p_c, l_c, N_ELEMENT_MODES, N_CRUNS);
  LIBXSMM_VLA_DECL(3, REALTYPE, l_p_c_asm, l_c_asm, N_ELEMENT_MODES, N_CRUNS);
  LIBXSMM_VLA_DECL(3, REALTYPE, l_p_c_gold, l_c_gold, N_ELEMENT_MODES, N_CRUNS);

  libxsmm_gemm_descriptor l_xgemm_desc;
#if defined(__EDGE_EXECUTE_F32__)
  libxsmm_smmfunction mykernel = NULL;
#else
  libxsmm_dmmfunction mykernel = NULL;
#endif

  struct timeval l_start, l_end;
  double l_total;

  if (argc != 4) {
    fprintf( stderr, "arguments: M #iters CSR-file!\n" );
    return -1;
  }

  /* touch B */
  for ( l_i = 0; l_i < N_QUANTITIES; l_i++) {
    for ( l_j = 0; l_j < N_ELEMENT_MODES; l_j++) {
      for ( l_k = 0; l_k < N_CRUNS; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_b, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS) = (REALTYPE)drand48();
      }
    }
  }

  /* touch C */
  for ( l_i = 0; l_i < N_QUANTITIES; l_i++) {
    for ( l_j = 0; l_j < N_ELEMENT_MODES; l_j++) {
      for ( l_k = 0; l_k < N_CRUNS; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_p_c,      l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS) = (REALTYPE)0.0;
        LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS) = (REALTYPE)0.0;
        LIBXSMM_VLA_ACCESS(3, l_p_c_asm,  l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS) = (REALTYPE)0.0;
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
            LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS)
              +=   l_a_de[(l_i*N_QUANTITIES)+l_jj]
                 * LIBXSMM_VLA_ACCESS(3, l_p_b, l_jj, l_j, l_k, N_ELEMENT_MODES, N_CRUNS);
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
            LIBXSMM_VLA_ACCESS(3, l_p_c, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS)
              +=   l_a_sp[l_rowstart+l_jj]
                 * LIBXSMM_VLA_ACCESS(3, l_p_b, l_colidx[l_rowptr[l_i] + l_jj], l_j, l_k, N_ELEMENT_MODES, N_CRUNS);
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
#if defined(__EDGE_EXECUTE_F32__)
  LIBXSMM_GEMM_DESCRIPTOR(l_xgemm_desc, LIBXSMM_GEMM_PRECISION_F32, 0/*flags*/,
    N_QUANTITIES, N_ELEMENT_MODES, N_QUANTITIES, 0, N_ELEMENT_MODES, N_ELEMENT_MODES,
    1.0, 1.0, LIBXSMM_PREFETCH_NONE);
  mykernel = libxsmm_create_xcsr_soa( &l_xgemm_desc, l_rowptr, l_colidx, (const void*)l_a_sp ).smm;
#else
  LIBXSMM_GEMM_DESCRIPTOR(l_xgemm_desc, LIBXSMM_GEMM_PRECISION_F64, 0/*flags*/,
    N_QUANTITIES, N_ELEMENT_MODES, N_QUANTITIES, 0, N_ELEMENT_MODES, N_ELEMENT_MODES,
    1.0, 1.0, LIBXSMM_PREFETCH_NONE);
  mykernel = libxsmm_create_xcsr_soa( &l_xgemm_desc, l_rowptr, l_colidx, (const void*)l_a_sp ).dmm;
#endif

  gettimeofday(&l_start, NULL);
  for ( l_n = 0; l_n < REPS; l_n++) {
    mykernel( l_a_sp, l_b, l_c_asm );
  }
  gettimeofday(&l_end, NULL);
  l_total = sec(l_start, l_end);
  printf("%fs for sparse (asm)\n", l_total);
  printf("%f GFLOPS for sparse (asm)\n", ((double)((double)REPS * (double)N_QUANTITIES * (double)l_elements * (double)N_CRUNS) * 2.0) / (l_total * 1.0e9));
  /* check for errors */
  for ( l_i = 0; l_i < N_QUANTITIES; l_i++) {
    for ( l_j = 0; l_j < N_ELEMENT_MODES; l_j++) {
      for ( l_k = 0; l_k < N_CRUNS; l_k++ ) {
        if (fabs( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS)
                    - LIBXSMM_VLA_ACCESS(3, l_p_c, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS) ) > l_max_error ) {
          l_max_error = fabs( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS)
                                - LIBXSMM_VLA_ACCESS(3, l_p_c, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS) );
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
        if (fabs( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS)
                    - LIBXSMM_VLA_ACCESS(3, l_p_c_asm, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS) ) > l_max_error ) {
          l_max_error = fabs( LIBXSMM_VLA_ACCESS(3, l_p_c_gold, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS)
                                -LIBXSMM_VLA_ACCESS(3, l_p_c_asm, l_i, l_j, l_k, N_ELEMENT_MODES, N_CRUNS) );
        }
      }
    }
  }
  printf("max error: %f\n", l_max_error);

  /* free */
  libxsmm_free( l_a_de );
  libxsmm_free( l_b );
  libxsmm_free( l_c );
  libxsmm_free( l_c_gold );
  libxsmm_free( l_c_asm );

  return 0;
}
