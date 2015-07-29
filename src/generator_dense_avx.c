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

void libxsmm_generator_dense_avx_load_C_MxN( libxsmm_generated_code*             io_generated_code,
                                             const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                             const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                             const libxsmm_xgemm_descriptor*     i_xgemm_desc,
                                             const unsigned int                  i_m_blocking,
                                             const unsigned int                  i_n_blocking ) {
#ifndef NDEGUG
  /* Do some test if it's possible to generated the requested code. 
     This is not done in release mode and therefore bad
     things might happen.... HUAAH */ 
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) || (i_m_blocking < 1) ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx_load_MxN, register blocking is invalid!!!\n");
    exit(-1);
  }
  /* test that l_m_blocking % i_micro_kernel_config->vector_length is 0 */
#endif

  /* deriving register blocking from kernel config */ 
  unsigned int l_m_blocking = i_m_blocking/i_micro_kernel_config->vector_length;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (i_n_blocking * l_m_blocking);

  /* load C accumulator */
  if (i_xgemm_desc->beta == 1) {
    /* adding to C, so let's load C */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        libxsmm_instruction_vec_move( io_generated_code, 
                                      i_micro_kernel_config->c_vmove_instruction, 
                                      i_gp_reg_mapping->gp_reg_c, 
                                      ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size), 
                                      i_micro_kernel_config->vector_name, 
                                      l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 0 );
      }
    }
  } else {
    /* overwriting C, so let's xout accumulator */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) { 
        libxsmm_instruction_vec_compute_reg( io_generated_code, 
                                             i_micro_kernel_config->vxor_instruction,
                                             i_micro_kernel_config->vector_name, 
                                             l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 
                                             l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 
                                             l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
    }
  }
}

void libxsmm_generator_dense_avx_store_C_MxN( libxsmm_generated_code*             io_generated_code,
                                              const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                              const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                              const libxsmm_xgemm_descriptor*     i_xgemm_desc,
                                              const unsigned int                  i_m_blocking,
                                              const unsigned int                  i_n_blocking ) {
  /* @TODO fix this test */ 
#ifndef NDEBUG
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx_store_MxN, i_n_blocking smaller 1 or larger 3!!!\n");
    exit(-1);
  }
  /* test that l_m_blocking % i_micro_kernel_config->vector_length is 0 */
#endif

  /* deriving register blocking from kernel config */ 
  unsigned int l_m_blocking = i_m_blocking/i_micro_kernel_config->vector_length;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (i_n_blocking * l_m_blocking);

  /* storing C accumulator */
  /* adding to C, so let's load C */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      libxsmm_instruction_vec_move( io_generated_code, 
                                    i_micro_kernel_config->c_vmove_instruction, 
                                    i_gp_reg_mapping->gp_reg_c, 
                                    ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size), 
                                    i_micro_kernel_config->vector_name, 
                                    l_vec_reg_acc_start + l_m + (l_m_blocking * l_n), 1 );
    }
  }

  if ( strcmp( i_xgemm_desc->prefetch, "BL2viaC" ) == 0 ) {
    /* determining how many prefetches we need M direction as we just need one prefetch per cache line */
    unsigned int l_m_advance = 64/((i_micro_kernel_config->vector_length) * (i_micro_kernel_config->datatype_size)); /* 64: hardcoded cache line length */
        
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      for (l_m = 0; l_m < l_m_blocking; l_m += l_m_advance ) {
        libxsmm_instruction_prefetch( io_generated_code, 
                                      i_micro_kernel_config->prefetch_instruction,
                                      i_gp_reg_mapping->gp_reg_b_prefetch, 
                                      ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size));
      }
    }
  }
}

void libxsmm_generator_dense_avx_compute_MxN( libxsmm_generated_code*             io_generated_code,
                                              const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                              const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                              const libxsmm_xgemm_descriptor*     i_xgemm_desc,
                                              const unsigned int                  i_m_blocking,
                                              const unsigned int                  i_n_blocking,
                                              const int                           i_offset ) {
  /* @TODO fix this test */
#ifndef NDEBUG
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx_compute_MxN, i_n_blocking smaller 1 or larger 3!!!\n");
    exit(-1);
  }
  /* test that l_m_blocking % i_micro_kernel_config->vector_length is 0 */
#endif
  /* deriving register blocking from kernel config */ 
  unsigned int l_m_blocking = i_m_blocking/i_micro_kernel_config->vector_length;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (i_n_blocking * l_m_blocking);

  /* broadcast from B -> into vec registers 0 to i_n_blocking */
  if ( i_offset != (-1) ) { 
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      libxsmm_instruction_vec_move( io_generated_code, 
                                    i_micro_kernel_config->b_vmove_instruction, 
                                    i_gp_reg_mapping->gp_reg_b, 
                                    ((i_micro_kernel_config->datatype_size) * i_offset) + (i_xgemm_desc->ldb * l_n * (i_micro_kernel_config->datatype_size)), 
                                    i_micro_kernel_config->vector_name, 
                                    l_n, 0 );
    }
  } else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      libxsmm_instruction_vec_move( io_generated_code, 
                                    i_micro_kernel_config->b_vmove_instruction, 
                                    i_gp_reg_mapping->gp_reg_b, 
                                    i_xgemm_desc->ldb * l_n *  i_micro_kernel_config->datatype_size, 
                                    i_micro_kernel_config->vector_name, 
                                    l_n, 0 );
    }
    libxsmm_instruction_alu_imm( io_generated_code,
                                 i_micro_kernel_config->alu_add_instruction, 
                                 i_gp_reg_mapping->gp_reg_b,
                                 i_micro_kernel_config->datatype_size );
  }

  /* load column vectors of A and multiply with all broadcasted row entries of B */
  for ( l_m = 0; l_m < l_m_blocking ; l_m++ ) {
    libxsmm_instruction_vec_move( io_generated_code, 
                                  i_micro_kernel_config->a_vmove_instruction, 
                                  i_gp_reg_mapping->gp_reg_a, 
                                  (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m, 
                                  i_micro_kernel_config->vector_name, 
                                  i_n_blocking, 0 );

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* post increment early */
      if ( (l_m == (l_m_blocking-1)) && (l_n == 0) ) {
        libxsmm_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a, 
                                     (i_xgemm_desc->lda)*(i_micro_kernel_config->datatype_size) );
      }
      /* issue fma / mul-add */
      /* @TODO add support for mul/add */
      libxsmm_instruction_vec_compute_reg( io_generated_code, 
                                           i_micro_kernel_config->vmul_instruction, 
                                           i_micro_kernel_config->vector_name, 
                                           i_n_blocking, 
                                           l_n, 
                                           l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
    }
  }
}

void libxsmm_generator_dense_avx_kernel( libxsmm_generated_code*        io_generated_code,
                                         const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                         const char*                     i_arch ) {
  /* define gp register mapping */
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_R9;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R10;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_R11;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14; 

  /* define the micro kernel code gen properties */
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_generator_dense_init_micro_kernel_config_fullvector( &l_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
  
  /* some hard coded parameters for k-blocking */
  unsigned int l_k_blocking = 4;
  unsigned int l_k_threshold = 30;

  /* initialize n-blocking */
  unsigned int l_n_done = 0;
  unsigned int l_n_done_old = 0;
  unsigned int l_n_blocking = 3;

  /* open asm */
  libxsmm_generator_dense_sse_avx_open_instruction_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );
  
  /* apply n_blocking */
  while (l_n_done != i_xgemm_desc->n) {
    l_n_done_old = l_n_done;
    l_n_done = l_n_done + (((i_xgemm_desc->n - l_n_done_old) / l_n_blocking) * l_n_blocking);

    if (l_n_done != l_n_done_old && l_n_done > 0) {

      libxsmm_generator_dense_header_nloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, l_n_blocking );
  
      unsigned int l_m_done = 0;
      unsigned int l_m_done_old = 0;
      unsigned int l_m_blocking = 16;

      /* apply m_blocking */
      while (l_m_done != i_xgemm_desc->m) {
        if (l_m_done == 0) {
          /* This is a SeisSol Order 6, HSW, DP performance fix */
          if ( (strcmp( i_arch, "hsw" ) == 0) && (i_xgemm_desc->single_precision == 0) ) { 
            l_m_done_old = l_m_done;
            if (i_xgemm_desc->m == 56) {
              l_m_done = 32;
            } else {
              l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
            }
          } else {
            l_m_done_old = l_m_done;
            l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
          }  
        } else {
          l_m_done_old = l_m_done;
          l_m_done = l_m_done + (((i_xgemm_desc->m - l_m_done_old) / l_m_blocking) * l_m_blocking);
        }
  
        if ( (l_m_done != l_m_done_old) && (l_m_done > 0) ) {
          libxsmm_generator_dense_header_mloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, l_m_blocking );
          libxsmm_generator_dense_avx_load_C_MxN( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_n_blocking );

          /* apply multiple k_blocking strategies */
          /* 1. we are larger the k_threshold and a multple of a predefined blocking parameter */
          if ((i_xgemm_desc->k % l_k_blocking) == 0 && i_xgemm_desc->k > l_k_threshold) {
            libxsmm_generator_dense_header_kloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, l_m_blocking, l_k_blocking);
            
            unsigned int l_k;
            for ( l_k = 0; l_k < l_k_blocking; l_k++) {
	      libxsmm_generator_dense_avx_compute_MxN(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, 
                                                      i_xgemm_desc, l_m_blocking, l_n_blocking, -1);
            }

            libxsmm_generator_dense_footer_kloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, i_xgemm_desc->k, 1 );
          } else {
            /* 2. we want to fully unroll below the threshold */
            if (i_xgemm_desc->k <= l_k_threshold) {
              unsigned int l_k;
              for ( l_k = 0; l_k < i_xgemm_desc->k; l_k++) {
	        libxsmm_generator_dense_avx_compute_MxN(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, 
                                                        i_xgemm_desc, l_m_blocking, l_n_blocking, l_k);
	      }
            /* 3. we are large than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling */
            } else {
	      unsigned int l_max_blocked_k = ((i_xgemm_desc->k)/l_k_blocking)*l_k_blocking;
	      if ( l_max_blocked_k > 0 ) {
	        libxsmm_generator_dense_header_kloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, l_m_blocking, l_k_blocking);
               
                unsigned int l_k;
                for ( l_k = 0; l_k < l_k_blocking; l_k++) {
	          libxsmm_generator_dense_avx_compute_MxN(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, 
                                                          i_xgemm_desc, l_m_blocking, l_n_blocking, -1);
                }

	        libxsmm_generator_dense_footer_kloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_max_blocked_k, 0 );
	      }
	      if (l_max_blocked_k > 0 ) {
                libxsmm_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, 
                                             l_gp_reg_mapping.gp_reg_b, l_max_blocked_k * l_micro_kernel_config.datatype_size );
	      }
              unsigned int l_k;
	      for ( l_k = l_max_blocked_k; l_k < i_xgemm_desc->k; l_k++) {
	        libxsmm_generator_dense_avx_compute_MxN(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, 
                                                        i_xgemm_desc, l_m_blocking, l_n_blocking, l_k);
	      }
            }
          }

          libxsmm_generator_dense_avx_store_C_MxN( io_generated_code, &l_gp_reg_mapping,  &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_n_blocking );
          libxsmm_generator_dense_footer_mloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_m_done);
        }

        /* switch to next smaller m_blocking */
        if (l_m_blocking == 2) {
          l_m_blocking = 1;
          libxsmm_generator_dense_init_micro_kernel_config_scalar( &l_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
        } else if (l_m_blocking == 4) {
          l_m_blocking = 2;
          libxsmm_generator_dense_init_micro_kernel_config_halfvector( &l_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
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

      libxsmm_generator_dense_footer_nloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_n_blocking);
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
  libxsmm_generator_dense_sse_avx_close_instruction_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );
}

void libxsmm_generator_dense_avx( libxsmm_generated_code*         io_generated_code,
                                  const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                  const char*                     i_arch ) {
  libxsmm_xgemm_descriptor l_xgemm_desc_mod = *i_xgemm_desc;
  unsigned int l_vector_length;

  /* determining vector length depending on architecture and precision */
  /* @TODO fix me */
  if ( (strcmp(i_arch, "wsm") == 0) && (l_xgemm_desc_mod.single_precision == 0) ) {
    l_vector_length = 2;
  } else if ( (strcmp(i_arch, "wsm") == 0) && (l_xgemm_desc_mod.single_precision == 1) ) {
    l_vector_length = 4;
  } else if ( (strcmp(i_arch, "snb") == 0) && (l_xgemm_desc_mod.single_precision == 0) ) {
    l_vector_length = 4;
  } else if ( (strcmp(i_arch, "snb") == 0) && (l_xgemm_desc_mod.single_precision == 1) ) {
    l_vector_length = 8;
  } else if ( (strcmp(i_arch, "hsw") == 0) && (l_xgemm_desc_mod.single_precision == 0) ) {
    l_vector_length = 4;
  } else if ( (strcmp(i_arch, "hsw") == 0) && (l_xgemm_desc_mod.single_precision == 1) ) {
    l_vector_length = 8;
  } else {
    fprintf(stderr, "received non-valid arch and precision in libxsmm_generator_dense_avx\n");
    exit(-1);
  }
 
  /* derive if alignment is possible */
  if ( (l_xgemm_desc_mod.lda % l_vector_length) == 0 ) {
    l_xgemm_desc_mod.aligned_a = 1;
  }
  if ( (l_xgemm_desc_mod.ldc % l_vector_length) == 0 ) {
    l_xgemm_desc_mod.aligned_c = 1;
  }

  /* enforce possible external overwrite */
  l_xgemm_desc_mod.aligned_a = l_xgemm_desc_mod.aligned_a && i_xgemm_desc->aligned_a;
  l_xgemm_desc_mod.aligned_c = l_xgemm_desc_mod.aligned_c && i_xgemm_desc->aligned_c;

  /* call actual kernel generation with revided parameters */
  libxsmm_generator_dense_avx_kernel(io_generated_code, &l_xgemm_desc_mod, i_arch );
}
