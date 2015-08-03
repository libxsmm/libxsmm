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

#include "generator_dense_sse3_avx_avx2_common.h"

void libxsmm_generator_dense_sse3_avx_avx2_load_C( libxsmm_generated_code*             io_generated_code,
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

void libxsmm_generator_dense_sse3_avx_avx2_store_C( libxsmm_generated_code*             io_generated_code,
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

unsigned int libxsmm_generator_dense_sse3_avx_avx2_get_inital_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                          const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                                                          const char*                     i_arch ) {
  unsigned int l_m_blocking = 0;

  if ( (strcmp( i_arch, "wsm" ) == 0) && (i_xgemm_desc->single_precision == 1) ) {
    l_m_blocking = 12;
  } else if ( (strcmp( i_arch, "wsm" ) == 0) && (i_xgemm_desc->single_precision == 0) ) {
    l_m_blocking = 6;
  } else if ( (strcmp( i_arch, "snb" ) == 0) && (i_xgemm_desc->single_precision == 1) ) {
    l_m_blocking = 24;
  } else if ( (strcmp( i_arch, "snb" ) == 0) && (i_xgemm_desc->single_precision == 0) ) {
    l_m_blocking = 12;
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && (i_xgemm_desc->single_precision == 1) ) {
    l_m_blocking = 32;
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && (i_xgemm_desc->single_precision == 0) ) {
    l_m_blocking = 16;
  } else {
    fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_avx2_get_inital_m_blocking unknown architecture!\n");
    exit(-1);
  }

  libxsmm_generator_dense_init_micro_kernel_config_fullvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );

  return l_m_blocking;
}

unsigned int libxsmm_generator_dense_sse3_avx_avx2_update_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                      const libxsmm_xgemm_descriptor* i_xgemm_desc, 
                                                                      const char*                     i_arch,
                                                                      const unsigned int              i_current_m_blocking ) {
  unsigned int l_m_blocking = 0;

  if ( (strcmp( i_arch, "wsm" ) == 0) && (i_xgemm_desc->single_precision == 1) ) {
    if (i_current_m_blocking == 4) {
      l_m_blocking = 1;
      libxsmm_generator_dense_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 12) {
      l_m_blocking = 8;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "wsm" ) == 0) && (i_xgemm_desc->single_precision == 0) ) {
    if (i_current_m_blocking == 2) {
      l_m_blocking = 1;
      libxsmm_generator_dense_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 4) {
      l_m_blocking = 2;
    } else if (i_current_m_blocking == 6) {
      l_m_blocking = 4;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "snb" ) == 0) && (i_xgemm_desc->single_precision == 1) ) {
    if (i_current_m_blocking == 4) {
      l_m_blocking = 1;
      libxsmm_generator_dense_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
      libxsmm_generator_dense_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 24) {
      l_m_blocking = 16;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "snb" ) == 0) && (i_xgemm_desc->single_precision == 0) ) {
    if (i_current_m_blocking == 2) {
      l_m_blocking = 1;
      libxsmm_generator_dense_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 4) {
      l_m_blocking = 2;
      libxsmm_generator_dense_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 12) {
      l_m_blocking = 8;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && (i_xgemm_desc->single_precision == 1) ) {
    if (i_current_m_blocking == 4) {
      l_m_blocking = 1;
      libxsmm_generator_dense_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
      libxsmm_generator_dense_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 24) {
      l_m_blocking = 16;
    } else if (i_current_m_blocking == 32) {
      l_m_blocking = 24;
    } else {
      /* we are done with m_blocking */
    }
  } else if ( (strcmp( i_arch, "hsw" ) == 0) && (i_xgemm_desc->single_precision == 0) ) {
    if (i_current_m_blocking == 2) {
      l_m_blocking = 1;
      libxsmm_generator_dense_init_micro_kernel_config_scalar( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 4) {
      l_m_blocking = 2;
      libxsmm_generator_dense_init_micro_kernel_config_halfvector( io_micro_kernel_config, i_xgemm_desc, i_arch, 0 );
    } else if (i_current_m_blocking == 8) {
      l_m_blocking = 4;
    } else if (i_current_m_blocking == 12) {
      l_m_blocking = 8;
    } else if (i_current_m_blocking == 16) {
      l_m_blocking = 12;
    } else {
      /* we are done with m_blocking */
    }
  } else {
    fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_avx2_update_m_blocking unknown architecture!\n");
    exit(-1);
  }

  return l_m_blocking;
}

