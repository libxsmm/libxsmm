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

#include "generator_dense_instructions.h"
#include "generator_dense_avx512_microkernel.h"

void libxsmm_generator_dense_avx512_microkernel( libxsmm_generated_code*             io_generated_code,
                                                 const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                 const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                 const libxsmm_xgemm_descriptor*     i_xgemm_desc,
                                                 const unsigned int                  i_n_blocking,
                                                 const unsigned int                  i_k_blocking,
                                                 const int                           i_offset ) {
#ifndef NDEBUG
  if ( i_n_blocking > 30 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_dense_avx512_microkernel: i_n_blocking exceeds 30\n");
    exit(-1); 
  }
  if ( (i_offset >= 0) && (i_k_blocking != 1) ) {
    fprintf(stderr, "LIBXSMM WARNING, libxsmm_generator_dense_avx512_microkernel: i_k_blocking is ignored as offset is >=0\n");
  }
#endif
  unsigned int l_n;
  unsigned int l_k;

  if (i_offset != (-1)) {
    libxsmm_instruction_vec_move( io_generated_code,
                                  i_micro_kernel_config->instruction_set,
                                  i_micro_kernel_config->a_vmove_instruction, 
                                  i_gp_reg_mapping->gp_reg_a, 
                                  i_xgemm_desc->lda * i_offset * i_micro_kernel_config->datatype_size, 
                                  i_micro_kernel_config->vector_name, 
                                  0, 
                                  i_micro_kernel_config->use_masking_a_c, 0 );

    /* current A prefetch, next 8 rows for the current column */
    if ( (strcmp( i_xgemm_desc->prefetch,"curAL2" ) == 0)         ||
         (strcmp( i_xgemm_desc->prefetch,"curAL2_BL2viaC" ) == 0)    ) {
      libxsmm_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a,
                                    (i_xgemm_desc->lda * i_offset * i_micro_kernel_config->datatype_size) + 64 );
    }

    /* next A prefetch "same" rows in "same" column, but in a different matrix */ 
    if ( (strcmp( i_xgemm_desc->prefetch,"AL2jpst" ) == 0)         ||
         (strcmp( i_xgemm_desc->prefetch,"AL2jpst_BL2viaC" ) == 0) ||
         (strcmp( i_xgemm_desc->prefetch,"AL2" ) == 0)             || 
         (strcmp( i_xgemm_desc->prefetch,"AL2_BL2viaC" ) == 0)        ) {
      libxsmm_instruction_prefetch( io_generated_code,
                                    i_micro_kernel_config->prefetch_instruction,
                                    i_gp_reg_mapping->gp_reg_a_prefetch,
                                    (i_xgemm_desc->lda * i_offset * i_micro_kernel_config->datatype_size) );
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_instruction_vec_compute_membcast( io_generated_code, 
                                                i_micro_kernel_config->instruction_set,
                                                i_micro_kernel_config->vmul_instruction,
                                                i_gp_reg_mapping->gp_reg_b,
                                                LIBXSMM_X86_GP_REG_UNDEF,
                                                0,
                                                (i_offset * i_micro_kernel_config->datatype_size) + (i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size * l_n),
                                                i_micro_kernel_config->vector_name,
                                                0,
                                                i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
    }
  } else {
    /* apply k blocking */
    for ( l_k = 0; l_k < i_k_blocking; l_k++ ) {
      if ( l_k == 0 ) {
        libxsmm_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->a_vmove_instruction, 
                                      i_gp_reg_mapping->gp_reg_a, 
                                      i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size, 
                                      i_micro_kernel_config->vector_name, 
                                      0, 
                                      i_micro_kernel_config->use_masking_a_c, 0 );
        if ( i_k_blocking > 1 ) {
          libxsmm_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->a_vmove_instruction, 
                                        i_gp_reg_mapping->gp_reg_a, 
                                        i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size, 
                                        i_micro_kernel_config->vector_name, 
                                        1, 
                                        i_micro_kernel_config->use_masking_a_c, 0 );
        }
      } else if ( l_k < (i_k_blocking - 1) ) {
        libxsmm_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->a_vmove_instruction, 
                                      i_gp_reg_mapping->gp_reg_a, 
                                      i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size, 
                                      i_micro_kernel_config->vector_name, 
                                      (l_k+1)%2, 
                                      i_micro_kernel_config->use_masking_a_c, 0 );
      }

      // current A prefetch, next 8 rows for the current column
      if ( (strcmp( i_xgemm_desc->prefetch, "curAL2" ) == 0)         ||
           (strcmp( i_xgemm_desc->prefetch, "curAL2_BL2viaC" ) == 0)    ) {
        libxsmm_instruction_prefetch( io_generated_code,
                                      i_micro_kernel_config->prefetch_instruction,
                                      i_gp_reg_mapping->gp_reg_a,
                                      (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) + 64 );
      }

      // next A prefetch "same" rows in "same" column, but in a different matrix 
      if ( (strcmp( i_xgemm_desc->prefetch, "AL2jpst" ) == 0)         || 
           (strcmp( i_xgemm_desc->prefetch, "AL2jpst_BL2viaC" ) == 0) ||
           (strcmp( i_xgemm_desc->prefetch, "AL2" ) == 0)             || 
           (strcmp( i_xgemm_desc->prefetch, "AL2_BL2viaC" ) == 0)        ) {
        libxsmm_instruction_prefetch( io_generated_code,
                                      i_micro_kernel_config->prefetch_instruction,
                                      i_gp_reg_mapping->gp_reg_a_prefetch,
                                      (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) );
        if ( l_k == (i_k_blocking - 1) ) {
          libxsmm_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction, 
                                       i_gp_reg_mapping->gp_reg_a_prefetch,
                                       i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
        }
      }

      if ( l_k == (i_k_blocking - 1) ) {
        libxsmm_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction, 
                                     i_gp_reg_mapping->gp_reg_a,
                                     i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }

      for ( l_n = 0; l_n < i_n_blocking; l_n++) {
        libxsmm_instruction_vec_compute_membcast( io_generated_code, 
                                                  i_micro_kernel_config->instruction_set,
                                                  i_micro_kernel_config->vmul_instruction,
                                                  i_gp_reg_mapping->gp_reg_b,
                                                  LIBXSMM_X86_GP_REG_UNDEF,
                                                  0,
                                                  (l_k * i_micro_kernel_config->datatype_size)+(i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size * l_n),
                                                  i_micro_kernel_config->vector_name,
                                                  l_k%2,
                                                  i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
      }
    }

    libxsmm_instruction_alu_imm( io_generated_code,
                                 i_micro_kernel_config->alu_add_instruction, 
                                 i_gp_reg_mapping->gp_reg_b,
                                 i_k_blocking * i_micro_kernel_config->datatype_size );
  }
}

