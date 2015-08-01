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
#include "generator_dense_sse_avx_avx2_common.h"
#include "generator_dense_avx2_microkernel.h"
#include "generator_dense_avx_microkernel.h"

void libxsmm_generator_dense_sse_avx_avx2_kernel( libxsmm_generated_code*         io_generated_code,
                                                  const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                                  const char*                     i_arch ) {
  /* define gp register mapping */
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* machting calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14; 

  /* set up architecture dependent compute micro kernel generator */
  void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*, 
                                  const libxsmm_xgemm_descriptor*, const unsigned int, const unsigned int, const int);
  if ( (strcmp(i_arch, "hsw") == 0) ) {
    l_generator_microkernel = libxsmm_generator_dense_avx2_microkernel;
  } else if ( (strcmp(i_arch, "snb") == 0) ) {
    l_generator_microkernel = libxsmm_generator_dense_avx_microkernel;
  } else {
    fprintf(stderr, "LIBXSMM ERROR libxsmm_generator_dense_sse_avx_avx2_kernel, cannot select microkernel\n");
    exit(-1);
  }

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
  
      /* define the micro kernel code gen properties, espically m-blocking affects the vector instruction length */
      unsigned int l_m_done = 0;
      unsigned int l_m_done_old = 0;
      unsigned int l_m_blocking = libxsmm_generator_dense_sse_avx_avx2_get_inital_m_blocking( &l_micro_kernel_config, i_xgemm_desc, i_arch );

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
          libxsmm_generator_dense_sse_avx_avx2_load_C( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_n_blocking );

          /* apply multiple k_blocking strategies */
          /* 1. we are larger the k_threshold and a multple of a predefined blocking parameter */
          if ((i_xgemm_desc->k % l_k_blocking) == 0 && i_xgemm_desc->k > l_k_threshold) {
            libxsmm_generator_dense_header_kloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, l_m_blocking, l_k_blocking);
            
            unsigned int l_k;
            for ( l_k = 0; l_k < l_k_blocking; l_k++) {
	      l_generator_microkernel(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, 
                                      i_xgemm_desc, l_m_blocking, l_n_blocking, -1);
            }

            libxsmm_generator_dense_footer_kloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, i_xgemm_desc->k, 1 );
          } else {
            /* 2. we want to fully unroll below the threshold */
            if (i_xgemm_desc->k <= l_k_threshold) {
              unsigned int l_k;
              for ( l_k = 0; l_k < i_xgemm_desc->k; l_k++) {
	        l_generator_microkernel(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, 
                                        i_xgemm_desc, l_m_blocking, l_n_blocking, l_k);
	      }
            /* 3. we are large than the threshold but not a multiple of the blocking factor -> largest possible blocking + remainder handling */
            } else {
	      unsigned int l_max_blocked_k = ((i_xgemm_desc->k)/l_k_blocking)*l_k_blocking;
	      if ( l_max_blocked_k > 0 ) {
	        libxsmm_generator_dense_header_kloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, l_m_blocking, l_k_blocking);
               
                unsigned int l_k;
                for ( l_k = 0; l_k < l_k_blocking; l_k++) {
	          l_generator_microkernel(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, 
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
	        l_generator_microkernel(io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, 
                                        i_xgemm_desc, l_m_blocking, l_n_blocking, l_k);
	      }
            }
          }

          libxsmm_generator_dense_sse_avx_avx2_store_C( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_n_blocking );
          libxsmm_generator_dense_footer_mloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_m_blocking, l_m_done );
        }

        /* switch to next smaller m_blocking */
        l_m_blocking = libxsmm_generator_dense_sse_avx_avx2_update_m_blocking( &l_micro_kernel_config, i_xgemm_desc, i_arch, l_m_blocking );
      }
      libxsmm_generator_dense_footer_nloop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc, l_n_blocking, l_n_done );
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
  libxsmm_generator_dense_sse_avx_avx2_kernel(io_generated_code, &l_xgemm_desc_mod, i_arch );
}
