/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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

#include "generator_gemm_sse3_microkernel.h"
#include "generator_x86_instructions.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_gemm_sse3_microkernel( libxsmm_generated_code*             io_generated_code,
                                               const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                               const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                               const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                               const unsigned int                 i_m_blocking,
                                               const unsigned int                 i_n_blocking,
                                               const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = i_m_blocking/i_micro_kernel_config->vector_length;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (i_n_blocking * l_m_blocking);

#if !defined(NDEBUG)
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  if ( i_m_blocking % i_micro_kernel_config->vector_length != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return;
  }
#endif

  if (l_m_blocking == 1) {
    /* load column vectors of A */
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                  i_micro_kernel_config->instruction_set,
                                  i_micro_kernel_config->a_vmove_instruction,
                                  i_gp_reg_mapping->gp_reg_a,
                                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                                  0,
                                  i_micro_kernel_config->vector_name,
                                  i_n_blocking, i_micro_kernel_config->use_masking_a_c, 0 );
    /* loop over columns of B */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* post increment of a pointer early */
      if ( l_n == 0 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     (i_xgemm_desc->lda)*(i_micro_kernel_config->datatype_size) );
      }
      /* different ways of using B */
      if ( i_offset != (-1) ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      ((i_micro_kernel_config->datatype_size) * i_offset) + (i_xgemm_desc->ldb * l_n * (i_micro_kernel_config->datatype_size)),
                                      i_micro_kernel_config->vector_name,
                                      l_n, i_micro_kernel_config->use_masking_a_c, 0 );
        /* generate shuffle as SSE3 has no broadcast load for single precision */
        if ( LIBXSMM_GEMM_PRECISION_F32 == i_xgemm_desc->datatype && ( i_micro_kernel_config->b_shuff_instruction != LIBXSMM_X86_INSTR_UNDEF ) ) {
          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->b_shuff_instruction,
                                               i_micro_kernel_config->vector_name,
                                               l_n,
                                               l_n,
                                               LIBXSMM_X86_VEC_REG_UNDEF,
                                               0 );
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      i_xgemm_desc->ldb * l_n *  i_micro_kernel_config->datatype_size,
                                      i_micro_kernel_config->vector_name,
                                      l_n, i_micro_kernel_config->use_masking_a_c, 0 );
        /* generate shuffle as SSE3 has no broadcast load for single precision */
        if ( LIBXSMM_GEMM_PRECISION_F32 == i_xgemm_desc->datatype && ( i_micro_kernel_config->b_shuff_instruction != LIBXSMM_X86_INSTR_UNDEF ) ) {
          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->b_shuff_instruction,
                                               i_micro_kernel_config->vector_name,
                                               l_n,
                                               l_n,
                                               LIBXSMM_X86_VEC_REG_UNDEF,
                                               0 );
        }
        if ( l_n == (i_n_blocking -1) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_b,
                                       i_micro_kernel_config->datatype_size );
        }
      }
      /* issue mul-add */
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vmul_instruction,
                                           i_micro_kernel_config->vector_name,
                                           i_n_blocking,
                                           l_n,
                                           LIBXSMM_X86_VEC_REG_UNDEF );
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vadd_instruction,
                                           i_micro_kernel_config->vector_name,
                                           l_n,
                                           l_vec_reg_acc_start + l_n,
                                           LIBXSMM_X86_VEC_REG_UNDEF );
    }
  } else {
    /* broadcast from B -> into vec registers 0 to i_n_blocking */
    if ( i_offset != (-1) ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      ((i_micro_kernel_config->datatype_size) * i_offset) + (i_xgemm_desc->ldb * l_n * (i_micro_kernel_config->datatype_size)),
                                      i_micro_kernel_config->vector_name,
                                      l_n, i_micro_kernel_config->use_masking_a_c, 0 );
        /* generate shuffle as SSE3 has no broadcast load for single precision */
        if ( LIBXSMM_GEMM_PRECISION_F32 == i_xgemm_desc->datatype && ( i_micro_kernel_config->b_shuff_instruction != LIBXSMM_X86_INSTR_UNDEF ) ) {
          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->b_shuff_instruction,
                                               i_micro_kernel_config->vector_name,
                                               l_n,
                                               l_n,
                                               LIBXSMM_X86_VEC_REG_UNDEF,
                                               0 );
        }
      }
    } else {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      i_xgemm_desc->ldb * l_n *  i_micro_kernel_config->datatype_size,
                                      i_micro_kernel_config->vector_name,
                                      l_n, i_micro_kernel_config->use_masking_a_c, 0 );
        /* generate shuffle as SSE3 has no broadcast load for single precision */
        if ( LIBXSMM_GEMM_PRECISION_F32 == i_xgemm_desc->datatype ) {
          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->b_shuff_instruction,
                                               i_micro_kernel_config->vector_name,
                                               l_n,
                                               l_n,
                                               LIBXSMM_X86_VEC_REG_UNDEF,
                                               0 );
        }
     }
     libxsmm_x86_instruction_alu_imm( io_generated_code,
                                  i_micro_kernel_config->alu_add_instruction,
                                  i_gp_reg_mapping->gp_reg_b,
                                  i_micro_kernel_config->datatype_size );
    }

    if (l_m_blocking == 3) {
      /* load column vectors of A and multiply with all broadcasted row entries of B */
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->a_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_a,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m,
                                      i_micro_kernel_config->vector_name,
                                      i_n_blocking, i_micro_kernel_config->use_masking_a_c, 0 );

        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          /* post increment early */
          if ( (l_m == (l_m_blocking-1)) && (l_n == 0) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a,
                                         (i_xgemm_desc->lda)*(i_micro_kernel_config->datatype_size) );
          }
          if (l_n < i_n_blocking - 1) {
            /* issued vmove to save loads from A */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 i_micro_kernel_config->a_vmove_instruction,
                                                 i_micro_kernel_config->vector_name,
                                                 i_n_blocking + l_n,
                                                 i_n_blocking + l_n + 1,
                                                 LIBXSMM_X86_VEC_REG_UNDEF );
          }
          /* issue mul+add */
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->vmul_instruction,
                                               i_micro_kernel_config->vector_name,
                                               l_n,
                                               i_n_blocking + l_n,
                                               LIBXSMM_X86_VEC_REG_UNDEF );
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->vadd_instruction,
                                               i_micro_kernel_config->vector_name,
                                               i_n_blocking + l_n,
                                               l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                                               LIBXSMM_X86_VEC_REG_UNDEF );
        }
      }
    } else {
      /* load column vectors of A and multiply with all broadcasted row entries of B */
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->a_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_a,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m,
                                      i_micro_kernel_config->vector_name,
                                      i_n_blocking + l_m, i_micro_kernel_config->use_masking_a_c, 0 );
      }
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          /* post increment early */
          if ( (l_m == (l_m_blocking-1)) && (l_n == 0) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a,
                                         (i_xgemm_desc->lda)*(i_micro_kernel_config->datatype_size) );
          }
          if (l_n < i_n_blocking - 1) {
            /* issued vmove to save loads from A */
            if (l_n == 0 ) {
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_micro_kernel_config->instruction_set,
                                                   i_micro_kernel_config->a_vmove_instruction,
                                                   i_micro_kernel_config->vector_name,
                                                   i_n_blocking + l_m + l_n,
                                                   i_n_blocking + l_m_blocking + l_n,
                                                   LIBXSMM_X86_VEC_REG_UNDEF );
            } else {
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_micro_kernel_config->instruction_set,
                                                   i_micro_kernel_config->a_vmove_instruction,
                                                   i_micro_kernel_config->vector_name,
                                                   i_n_blocking + l_m_blocking + l_n - 1,
                                                   i_n_blocking + l_m_blocking + l_n,
                                                   LIBXSMM_X86_VEC_REG_UNDEF );
            }
          }
          /* issue mul/add */
          if (l_n == 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 i_micro_kernel_config->vmul_instruction,
                                                 i_micro_kernel_config->vector_name,
                                                 l_n,
                                                 i_n_blocking + l_m + l_n,
                                                 LIBXSMM_X86_VEC_REG_UNDEF );
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 i_micro_kernel_config->vadd_instruction,
                                                 i_micro_kernel_config->vector_name,
                                                 i_n_blocking + l_m + l_n,
                                                 l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) ,
                                                 LIBXSMM_X86_VEC_REG_UNDEF );
          } else {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 i_micro_kernel_config->vmul_instruction,
                                                 i_micro_kernel_config->vector_name,
                                                 l_n,
                                                 i_n_blocking + l_m_blocking + l_n - 1,
                                                 LIBXSMM_X86_VEC_REG_UNDEF );
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 i_micro_kernel_config->vadd_instruction,
                                                 i_micro_kernel_config->vector_name,
                                                 i_n_blocking + l_m_blocking + l_n - 1,
                                                 l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) ,
                                                 LIBXSMM_X86_VEC_REG_UNDEF );
          }
        }
      }
    }
  }
}

