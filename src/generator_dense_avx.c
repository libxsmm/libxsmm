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
#ifndef NDEGUG
  /* Do some test if it's possible to generated the requested code. 
     This is not done in release mode and therefore bad
     things might happen.... HUAAH */ 
  if ( (i_n_blocking * i_m_blocking > 12) || (i_n_blocking < 1) || (i_m_blocking < 1) ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx_load_MxN, register blocking is invalid!!!\n");
    exit(-1);
  }
#endif

  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (i_n_blocking * i_m_blocking);

  /* load C accumulator */
  if (i_beta == 1) {
    /* adding to C, so let's load C */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < i_m_blocking; l_m++ ) {
        libxsmm_instruction_vec_move( io_generated_code, 
                                      i_vload_instr, i_gp_reg_load, ((l_n * i_ldc) + (l_m * i_vector_length)) * i_datatype_size, i_vector_name, l_vec_reg_acc_start + l_m + (i_m_blocking * l_n), 0 );
      }
    }
  } else {
    /* overwriting C, so let's xout accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < i_m_blocking; l_m++ ) { 
        libxsmm_instruction_vec_compute_reg( io_generated_code, 
                                             i_vxor_instr, i_vector_name, l_vec_reg_acc_start + l_m + (i_m_blocking * l_n), l_vec_reg_acc_start + l_m + (i_m_blocking * l_n), l_vec_reg_acc_start + l_m + (i_m_blocking * l_n) );
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
#ifndef NDEBUG
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx_store_MxN, i_n_blocking smaller 1 or larger 3!!!\n");
    exit(-1);
  }
#endif

  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (i_n_blocking * i_m_blocking);

  /* storing C accumulator */
  /* adding to C, so let's load C */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < i_m_blocking; l_m++ ) {
      libxsmm_instruction_vec_move( io_generated_code, 
                                    i_vstore_instr, i_gp_reg_store, ((l_n * i_ldc) + (l_m * i_vector_length)) * i_datatype_size, i_vector_name, l_vec_reg_acc_start + l_m + (i_m_blocking * l_n), 1 );
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
#ifndef NDEBUG
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx_compute_MxN, i_n_blocking smaller 1 or larger 3!!!\n");
    exit(-1);
  }
#endif

  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (i_n_blocking * i_m_blocking);

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
                                           i_vmul_instr, i_vector_name, i_n_blocking, l_n, l_vec_reg_acc_start + l_m + (i_m_blocking * l_n) );
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

  unsigned int l_gp_reg_a = LIBXSMM_X86_GP_REG_R9;
  unsigned int l_gp_reg_b = LIBXSMM_X86_GP_REG_R8;
  unsigned int l_gp_reg_c = LIBXSMM_X86_GP_REG_R10;
  unsigned int l_gp_reg_pre_a = LIBXSMM_X86_GP_REG_R11;
  unsigned int l_gp_reg_pre_b = LIBXSMM_X86_GP_REG_R12;
  unsigned int l_gp_reg_mloop = LIBXSMM_X86_GP_REG_R14;
  unsigned int l_gp_reg_nloop = LIBXSMM_X86_GP_REG_R15;
  unsigned int l_gp_reg_kloop = LIBXSMM_X86_GP_REG_R13;
  
  /* open asm */
  libxsmm_generator_dense_sse_avx_open_kernel( io_generated_code, l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_pre_b,
                                               l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop, i_prefetch);
  
  /* apply n_blocking */
  while (l_n_done != i_n) {
    l_n_done_old = l_n_done;
    l_n_done = l_n_done + (((i_n - l_n_done_old) / l_n_blocking) * l_n_blocking);

    if (l_n_done != l_n_done_old && l_n_done > 0) {

      libxsmm_generator_dense_header_nloop( io_generated_code, l_gp_reg_mloop, l_gp_reg_nloop, l_n_blocking );
  
      unsigned int l_k_blocking = 4;
      unsigned int l_k_threshold = 30;
      unsigned int l_m_done = 0;
      unsigned int l_m_done_old = 0;
      unsigned int l_m_blocking = 12;

      /* apply m_blocking */
      while (l_m_done != i_m) {
#if 0
        if (mDone == 0) { 
          mDone_old = mDone;
          if (M == 56) {
            mDone = 32;
          } else {
            mDone = mDone + (((M - mDone_old) / m_blocking) * m_blocking);
          }  
        } else {
#endif
        /* @TODO enable upper part again later */
        l_m_done_old = l_m_done;
        l_m_done = l_m_done + (((i_m - l_m_done_old) / l_m_blocking) * l_m_blocking);
  
        if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
          libxsmm_generator_dense_header_mloop( io_generated_code, l_gp_reg_mloop, l_m_blocking );
          libxsmm_generator_dense_avx_load_C_MxN( io_generated_code, l_c_vmove_instr, l_gp_reg_c, l_vxor_instr, i_vector_name,
                                                  l_m_blocking/i_vector_length, l_n_blocking, i_ldc, i_beta, i_vector_length, l_datatype_size);

          /* apply multiple k_blocking strategies */
          /* 1. we are larger the k_threshold and a multple of a predefined blocking parameter */
          if ((i_k % l_k_blocking) == 0 && i_k > l_k_threshold) {
            libxsmm_generator_dense_header_kloop( io_generated_code, l_gp_reg_kloop, l_m_blocking, l_k_blocking);
            
            unsigned int l_k;
            for ( l_k = 0; l_k < l_k_blocking; l_k++) {
	      libxsmm_generator_dense_avx_compute_MxN(io_generated_code, l_a_vmove_instr, l_b_vmove_instr, l_alu_add_instr, l_gp_reg_a, l_gp_reg_b, 
                                                      l_vmul_instr, l_vadd_instr, i_vector_name, l_m_blocking/i_vector_length, l_n_blocking,
                                                      i_lda, i_ldb, i_vector_length, l_datatype_size, -1);
            }

            libxsmm_generator_dense_footer_kloop( io_generated_code, l_gp_reg_kloop, l_gp_reg_b, l_m_blocking, i_k, l_datatype_size, 1 );
          } else {
            /* 2. we want to fully unroll below the threshold */
            if (i_k <= l_k_threshold) {
              unsigned int l_k;
              for ( l_k = 0; l_k < i_k; l_k++) {
	        libxsmm_generator_dense_avx_compute_MxN(io_generated_code, l_a_vmove_instr, l_b_vmove_instr, l_alu_add_instr, l_gp_reg_a, l_gp_reg_b, 
                                                        l_vmul_instr, l_vadd_instr, i_vector_name, l_m_blocking/i_vector_length, l_n_blocking,
                                                        i_lda, i_ldb, i_vector_length, l_datatype_size, l_k);
	      }
            /* 3. we are large than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling */
            } else {
	      unsigned int l_max_blocked_k = (i_k/l_k_blocking)*l_k_blocking;
	      if ( l_max_blocked_k > 0 ) {
	        libxsmm_generator_dense_header_kloop( io_generated_code, l_gp_reg_kloop, l_m_blocking, l_k_blocking);
               
                unsigned int l_k;
                for ( l_k = 0; l_k < l_k_blocking; l_k++) {
	          libxsmm_generator_dense_avx_compute_MxN(io_generated_code, l_a_vmove_instr, l_b_vmove_instr, l_alu_add_instr, l_gp_reg_a, l_gp_reg_b, 
                                                          l_vmul_instr, l_vadd_instr, i_vector_name, l_m_blocking/i_vector_length, l_n_blocking,
                                                          i_lda, i_ldb, i_vector_length, l_datatype_size, -1);
                }

	        libxsmm_generator_dense_footer_kloop( io_generated_code, l_gp_reg_kloop, l_gp_reg_b, l_m_blocking, i_k, l_datatype_size, 0 );
	      }
	      if (l_max_blocked_k > 0 ) {
                libxsmm_instruction_alu_imm( io_generated_code, "subq", l_gp_reg_b, l_max_blocked_k * l_datatype_size );
	      }
              unsigned int l_k;
	      for ( l_k = l_max_blocked_k; l_k < i_k; l_k++) {
	        libxsmm_generator_dense_avx_compute_MxN(io_generated_code, l_a_vmove_instr, l_b_vmove_instr, l_alu_add_instr, l_gp_reg_a, l_gp_reg_b, 
                                                        l_vmul_instr, l_vadd_instr, i_vector_name, l_m_blocking/i_vector_length, l_n_blocking,
                                                        i_lda, i_ldb, i_vector_length, l_datatype_size, l_k);
	      }
            }
          }

          libxsmm_generator_dense_avx_store_C_MxN( io_generated_code, l_c_vmove_instr, l_gp_reg_c, l_prefetch_instr, l_gp_reg_pre_b, i_vector_name,
                                                   i_prefetch, l_m_blocking/i_vector_length, l_n_blocking, i_ldc, i_vector_length, l_datatype_size);
          libxsmm_generator_dense_footer_mloop( io_generated_code, l_gp_reg_a, l_gp_reg_c, l_gp_reg_mloop, l_m_blocking,
                                                i_m, i_k, i_lda, i_prefetch, l_gp_reg_pre_b, l_datatype_size);
        }

        /* switch to next smaller m_blocking */
        if (l_m_blocking == 2) {
          l_m_blocking = 1;
        } else if (l_m_blocking == 4) {
          l_m_blocking = 2;
        } else if (l_m_blocking == 8) {
          l_m_blocking = 4;
        } else if (l_m_blocking == 12) {
          l_m_blocking = 8;
        } else if (l_m_blocking == 16) {
          l_m_blocking = 12;
        } else {
          /* we are done with m_blocking */
        }
      }

      libxsmm_generator_dense_footer_nloop( io_generated_code, l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_nloop, l_n_blocking,
                                            i_m, i_n, i_ldb, i_ldc, i_prefetch, l_gp_reg_pre_b, l_datatype_size);
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
  libxsmm_generator_dense_sse_avx_close_kernel( io_generated_code, l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_pre_b,
                                                l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop, i_prefetch);
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
