/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#include "generator_common.h"
#include "generator_dense_common.h"
#include "generator_dense_instructions.h"

void libxsmm_generator_dense_avx_load_C_MxN(char**             io_generated_code,
                                            const char*        i_vload_instr,
                                            const unsigned int i_gp_reg_load,
                                            const char*        i_vxor_instr,
                                            const char*        i_vector_name,
                                            const unsigned int i_m_blocking,
                                            const unsigned int i_n_blocking,
                                            const unsigned int i_ldc,
                                            const int          i_beta,
                                            const unsigned int i_vector_length,
                                            const unsigned int i_datatype_size) {
  /* @TODO fix this test */ 
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx_load_MxN, i_n_blocking smaller 1 or larger 3!!!\n");
    exit(-1);
  }

  unsigned int l_n = 0;
  unsigned int l_m = 0;

  /* C accumulator has registers xmm/ymm7-15 */
  if (i_beta == 1) {
    /* adding to C, so let's load C */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < i_m_blocking; l_m++ ) {
        libxsmm_instruction_vec_move( io_generated_code, 
                                      i_vload_instr, i_gp_reg_load, ((l_n * i_ldc) + (l_m * i_vector_length)) * i_datatype_size, i_vector_name, 7 + l_m + (i_m_blocking * l_n), 0 );
      }
    }
  } else {
    /* overwriting C, so let's xout accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < i_m_blocking; l_m++ ) { 
        libxsmm_instruction_vec_compute_reg( io_generated_code, 
                                             i_vxor_instr, i_vector_name, 7 + l_m + (i_m_blocking * l_n), 7 + l_m + (i_m_blocking * l_n), 7 + l_m + (i_m_blocking * l_n) );
      }
    }
  }
}

void libxsmm_generator_dense_avx_store_C_MxN(char** io_generated_code,
                                             const char* i_vstore_instr,
                                             const unsigned int i_gp_reg_store,
                                             const char* i_prefetch_instr,
                                             const unsigned int i_gp_reg_prefetch,
                                             const char* i_vector_name,
                                             const char* i_prefetch,
                                             const unsigned int i_m_blocking,
                                             const unsigned int i_n_blocking,
                                             const unsigned int i_ldc,
                                             const unsigned int i_vector_length,
                                             const unsigned int i_datatype_size) {
  /* @TODO fix this test */ 
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx_store_MxN, i_n_blocking smaller 1 or larger 3!!!\n");
    exit(-1);
  }

  unsigned int l_n = 0;
  unsigned int l_m = 0;

  /* C accumulator has registers xmm/ymm7-15 */
  /* adding to C, so let's load C */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < i_m_blocking; l_m++ ) {
      libxsmm_instruction_vec_move( io_generated_code, 
                                    i_vstore_instr, i_gp_reg_store, ((l_n * i_ldc) + (l_m * i_vector_length)) * i_datatype_size, i_vector_name, 7 + l_m + (i_m_blocking * l_n), 1 );
    }
  }

  if ( strcmp( i_prefetch, "BL2viaC" ) == 0 ) {
    /* determining how many prefetches we need M direction as we just need one prefetch per cache line */
    unsigned int l_m_advance = 64/(i_vector_length * i_datatype_size); /* 64: hardcoded cache line length */
        
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      for (l_m = 0; l_m < i_m_blocking; l_m += l_m_advance ) {
        libxsmm_instruction_prefetch( io_generated_code, 
                                      i_prefetch_instr, i_gp_reg_prefetch, ((l_n * i_ldc) + (l_m * i_vector_length)) * i_datatype_size);
      }
    }
  }
}

void libxsmm_generator_dense_avx_compute_MxN(char** io_generated_code,
                                             const char* i_a_load_instr,
                                             const char* i_b_load_instr,
                                             const char* i_add_instr,
                                             const unsigned int i_gp_reg_a,
                                             const unsigned int i_gp_reg_b,
                                             const char* i_vmul_instr,
                                             const char* i_vadd_instr,                                          
                                             const char* i_vector_name,
                                             const unsigned int i_m_blocking,
                                             const unsigned int i_n_blocking,
                                             const unsigned int i_lda,
                                             const unsigned int i_ldb,
                                             const unsigned int i_vector_length,
                                             const unsigned int i_datatype_size,
                                             const int          i_offset) {
  /* @TODO fix this test */
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx_compute_MxN, i_n_blocking smaller 1 or larger 3!!!\n");
    exit(-1);
  }

  unsigned int l_n;
  unsigned int l_m;

  /* broadcast from B -> into vec registers 0 to i_n_blocking */
  if ( i_offset != (-1) ) { 
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      libxsmm_instruction_vec_move( io_generated_code, 
                                    i_b_load_instr, i_gp_reg_b, (i_datatype_size * i_offset) + (i_ldb * l_n * i_datatype_size), i_vector_name, l_n, 0 );
    }
  } else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      libxsmm_instruction_vec_move( io_generated_code, 
                                    i_b_load_instr, i_gp_reg_b, i_ldb * l_n * i_datatype_size, i_vector_name, l_n, 0 );
    }
    libxsmm_instruction_alu_imm( io_generated_code,
                                 i_add_instr, i_gp_reg_b, i_datatype_size );
  }

  /* load column vectors of A and multiply with all broadcasted row entries of B */
  for ( l_m = 0; l_m < i_m_blocking ; l_m++ ) {
    libxsmm_instruction_vec_move( io_generated_code, 
                                  i_a_load_instr, i_gp_reg_a, i_datatype_size * i_vector_length * l_m, i_vector_name, i_n_blocking, 0 );
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* post increment early */
      if ( (l_m == (i_m_blocking-1)) && (l_n == 0) ) {
        libxsmm_instruction_alu_imm( io_generated_code,
                                     i_add_instr, i_gp_reg_a, i_lda*i_datatype_size );
      }
      /* issue fma / mul-add */
      /* @TODO add support for mul/add */
      libxsmm_instruction_vec_compute_reg( io_generated_code, 
                                           i_vmul_instr, i_vector_name, i_n_blocking, l_n, 7 + l_m + (i_m_blocking * l_n) );
    }
  }
}

void libxsmm_generator_dense_avx_kernel(char**             io_generated_code,
                                        const unsigned int i_m,
                                        const unsigned int i_n,
                                        const unsigned int i_k,
                                        const unsigned int i_lda,
                                        const unsigned int i_ldb,
                                        const unsigned int i_ldc, 
                                        const int          i_alpha,
                                        const int          i_beta,
                                        const unsigned int i_aligned_a,
                                        const unsigned int i_aligned_c,
                                        const char*        i_arch,
                                        const char*        i_prefetch,
                                        const unsigned int i_single_precision,
                                        const unsigned int i_vector_length,
                                        const char*        i_vector_name) {
  /* functions pointers such that we can handle different m_blockings dynamically */
  void (*l_generatorLoad)   (char** io_generated_code, int, int, int, int);
  void (*l_generatorStore)  (char** io_generated_code, int, int, int, char*);
  void (*l_generatorCompute)(char** io_generated_code, int, int, int, int, int, int, int);
  unsigned int l_n_done = 0;
  unsigned int l_n_done_old = 0;
  unsigned int l_n_blocking = 3;

  char* l_c_vmove_instr = "vmovapd";
  char* l_a_vmove_instr = "vmovapd";
  char* l_vxor_instr = "vxorpd";
  char* l_prefetch_instr = "prefetch1";
  char* l_b_vmove_instr = "vbroadcastsd";
  char* l_alu_add_instr = "addq";
  char* l_vmul_instr = "vfmadd231pd";
  char* l_vadd_instr = NULL;
  unsigned int l_datatype_size = 8;
  
  /* open asm */
  libxsmm_generator_dense_sse_avx_open_kernel( io_generated_code,
                                               i_prefetch);

  libxsmm_generator_dense_avx_load_C_MxN( io_generated_code,
                                          l_c_vmove_instr,
                                        10,
                                        l_vxor_instr,
                                        i_vector_name,
                                        3,
                                        l_n_blocking,
                                        i_ldc,
                                        i_beta,
                                        i_vector_length,
                                        l_datatype_size);

  libxsmm_generator_dense_avx_store_C_MxN( io_generated_code,
                                        l_c_vmove_instr,
                                        10,
                                        l_prefetch_instr,
                                        12,
                                        i_vector_name,
                                        i_prefetch,
                                        3,
                                        l_n_blocking,
                                        i_ldc,
                                        i_vector_length,
                                        l_datatype_size);

  libxsmm_generator_dense_avx_compute_MxN(io_generated_code,
                                          l_a_vmove_instr,
                                          l_b_vmove_instr,
                                          l_alu_add_instr,
                                          9,
                                          8,
                                          l_vmul_instr,
                                          l_vadd_instr,                                          
                                          i_vector_name,
                                          3,
                                          l_n_blocking,
                                          i_lda,
                                          i_ldb,
                                          i_vector_length,
                                          l_datatype_size,
                                          -1);
  
  /* apply n_blocking */
  while (l_n_done != i_n) {
    l_n_done_old = l_n_done;
    l_n_done = l_n_done + (((i_n - l_n_done_old) / l_n_blocking) * l_n_blocking);

    if (l_n_done != l_n_done_old && l_n_done > 0) {
#if 0
      header_nloop_dp_asm(codestream, n_blocking);
  
      int k_blocking = 4;
      int k_threshold = 30;
      int mDone = 0;
      int mDone_old = 0;
      int m_blocking = 16;

      // apply m_blocking
      while (mDone != M) {
        if (mDone == 0) {
          mDone_old = mDone;
          if (M == 56) {
            mDone = 32;
          } else {
            mDone = mDone + (((M - mDone_old) / m_blocking) * m_blocking);
          }  
        } else {
          mDone_old = mDone;
          mDone = mDone + (((M - mDone_old) / m_blocking) * m_blocking);
        }

        // switch to a different m_blocking
        if (m_blocking == 16) {
          l_generatorLoad = &avx2_load_16xN_dp_asm;
          l_generatorStore = &avx2_store_16xN_dp_asm;
          l_generatorCompute = &avx2_kernel_16xN_dp_asm;
        } else if (m_blocking == 12) {
          l_generatorLoad = &avx_load_12xN_dp_asm;
          l_generatorStore = &avx_store_12xN_dp_asm;
          l_generatorCompute = &avx2_kernel_12xN_dp_asm;
        } else if (m_blocking == 8) {
          l_generatorLoad = &avx_load_8xN_dp_asm;
          l_generatorStore = &avx_store_8xN_dp_asm;
          l_generatorCompute = &avx2_kernel_8xN_dp_asm;
        } else if (m_blocking == 4) {
          l_generatorLoad = &avx_load_4xN_dp_asm;
          l_generatorStore = &avx_store_4xN_dp_asm;
          l_generatorCompute = &avx2_kernel_4xN_dp_asm;
        } else if (m_blocking == 2) {
          l_generatorLoad = &avx_load_2xN_dp_asm;
          l_generatorStore = &avx_store_2xN_dp_asm;
          l_generatorCompute = &avx2_kernel_2xN_dp_asm;
        } else if (m_blocking == 1) {
          l_generatorLoad = &avx_load_1xN_dp_asm;
          l_generatorStore = &avx_store_1xN_dp_asm;
          l_generatorCompute = &avx2_kernel_1xN_dp_asm;      
        } else {
          std::cout << " !!! ERROR, avx2_generate_kernel_dp, m_blocking is out of range!!! " << std::endl;
          exit(-1);
        }

        if (mDone != mDone_old && mDone > 0) {
          header_mloop_dp_asm(codestream, m_blocking);
          (*l_generatorLoad)(codestream, ldc, alignC, i_beta, n_blocking);

          if ((K % k_blocking) == 0 && K > k_threshold) {
            header_kloop_dp_asm(codestream, m_blocking, k_blocking);

            for (int k = 0; k < k_blocking; k++) {
	      (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, -1, n_blocking);
            }

            footer_kloop_dp_asm(codestream, m_blocking, K);
          } else {
            // we want to fully unroll
            if (K <= k_threshold) {
              for (int k = 0; k < K; k++) {
	        (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, k, n_blocking);
	      }
            } else {
	      // we want to block, but K % k_blocking != 0
	      int max_blocked_K = (K/k_blocking)*k_blocking;
	      if (max_blocked_K > 0 ) {
	        header_kloop_dp_asm(codestream, m_blocking, k_blocking);
	        for (int k = 0; k < k_blocking; k++) {
	          (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, -1, n_blocking);
	        }
	        footer_kloop_notdone_dp_asm(codestream, m_blocking, max_blocked_K );
	      }
	      if (max_blocked_K > 0 ) {
	        codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
	      }
	      for (int k = max_blocked_K; k < K; k++) {
	        (*l_generatorCompute)(codestream, lda, ldb, ldc, alignA, alignC, k, n_blocking);
	      }
            }
          }

          (*l_generatorStore)(codestream, ldc, alignC, n_blocking, tPrefetch);
          footer_mloop_dp_asm(codestream, m_blocking, K, mDone, lda, tPrefetch);
        }

        // switch to a different m_blocking
        if (m_blocking == 2) {
          m_blocking = 1;
        } else if (m_blocking == 4) {
          m_blocking = 2;
        } else if (m_blocking == 8) {
          m_blocking = 4;
        } else if (m_blocking == 12) {
          m_blocking = 8;
        } else if (m_blocking == 16) {
          m_blocking = 12;
        } else {
          // we are done with m_blocking
        }
      }

      footer_nloop_dp_asm(codestream, n_blocking, nDone, M, lda, ldb, ldc, tPrefetch);
#endif
    }

    /* switch to a different, smaller n_blocking */
    if (l_n_blocking == 2) {
      l_n_blocking = 1;
    } else if (l_n_blocking == 3) {
      l_n_blocking = 2;
    } else {
      /* we are done with n_blocking */ 
    }
  }

  /* close asm */
  libxsmm_generator_dense_sse_avx_close_kernel( io_generated_code,
                                                i_prefetch);
}

void libxsmm_generator_dense_avx(char**             io_generated_code,
                                 const unsigned int i_m,
                                 const unsigned int i_n,
                                 const unsigned int i_k,
                                 const unsigned int i_lda,
                                 const unsigned int i_ldb,
                                 const unsigned int i_ldc, 
                                 const int          i_alpha,
                                 const int          i_beta,
                                 const unsigned int i_aligned_a,
                                 const unsigned int i_aligned_c,
                                 const char*        i_arch,
                                 const char*        i_prefetch,
                                 const unsigned int i_single_precision) {
  unsigned int l_vector_length = 0;
  unsigned int l_aligned_a = 0;
  unsigned int l_aligned_c = 0;
  char* l_vector_name = NULL;

  /* determining vector length depending on architecture and precision */
  if ( (strcmp(i_arch, "wsm") == 0) && (i_single_precision == 0) ) {
    l_vector_length = 2;
    l_vector_name = "xmm";
  } else if ( (strcmp(i_arch, "wsm") == 0) && (i_single_precision == 1) ) {
    l_vector_length = 4;
    l_vector_name = "xmm";
  } else if ( (strcmp(i_arch, "snb") == 0) && (i_single_precision == 0) ) {
    l_vector_length = 4;
    l_vector_name = "ymm";
  } else if ( (strcmp(i_arch, "snb") == 0) && (i_single_precision == 1) ) {
    l_vector_length = 8;
    l_vector_name = "ymm";
  } else if ( (strcmp(i_arch, "hsw") == 0) && (i_single_precision == 0) ) {
    l_vector_length = 4;
    l_vector_name = "ymm";
  } else if ( (strcmp(i_arch, "hsw") == 0) && (i_single_precision == 1) ) {
    l_vector_length = 8;
    l_vector_name = "ymm";
  } else {
    fprintf(stderr, "received non-valid arch and precsoin in libxsmm_generator_dense_sse_avx\n");
    exit(-1);
  }
 
  /* derive if alignment is possible */
  if ( (i_lda % l_vector_length) == 0 ) {
    l_aligned_a = 1;
  }
  if ( (i_ldc % l_vector_length) == 0 ) {
    l_aligned_c = 1;
  }

  /* enforce possible external overwrite */
  l_aligned_a = l_aligned_a && i_aligned_a;
  l_aligned_c = l_aligned_c && i_aligned_c;

  /* call actual kernel generation with revided parameters */
  libxsmm_generator_dense_avx_kernel(io_generated_code,
                                     i_m, i_n, i_k,
                                     i_lda, i_ldb, i_ldc, 
                                     i_alpha, i_beta,
                                     l_aligned_a, l_aligned_c,
                                     i_arch,
                                     i_prefetch,
                                     i_single_precision,
                                     l_vector_length, l_vector_name);

}
