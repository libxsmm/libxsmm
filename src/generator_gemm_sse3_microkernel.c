/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_gemm_sse3_microkernel.h"
#include "generator_x86_instructions.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
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
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;

  /* check that m_blocking is a multiple of vlen and that n_blocking is valid */
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  if ( i_m_blocking % i_micro_kernel_config->vector_length != 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return;
  }

  if (l_m_blocking == 1) {
    /* load column vectors of A */
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                  i_micro_kernel_config->instruction_set,
                                  i_micro_kernel_config->a_vmove_instruction,
                                  i_gp_reg_mapping->gp_reg_a,
                                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                                  0,
                                  i_micro_kernel_config->vector_name,
                                  i_n_blocking, 0, 1, 0 );
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
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size);
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        /* generate shuffle as SSE3 has no broadcast load for single precision */
        if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) && ( i_micro_kernel_config->b_shuff_instruction != LIBXSMM_X86_INSTR_UNDEF ) ) {
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
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        /* generate shuffle as SSE3 has no broadcast load for single precision */
        if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) && ( i_micro_kernel_config->b_shuff_instruction != LIBXSMM_X86_INSTR_UNDEF ) ) {
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
          /* handle trans B */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size;
          } else {
            l_b_offset = i_micro_kernel_config->datatype_size;
          }

          libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_b,
                                       l_b_offset );
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
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size);
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        /* generate shuffle as SSE3 has no broadcast load for single precision */
        if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) && ( i_micro_kernel_config->b_shuff_instruction != LIBXSMM_X86_INSTR_UNDEF ) ) {
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
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        /* generate shuffle as SSE3 has no broadcast load for single precision */
        if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
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
      /* handle trans B */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size;
      } else {
        l_b_offset = i_micro_kernel_config->datatype_size;
      }

      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                  i_micro_kernel_config->alu_add_instruction,
                                  i_gp_reg_mapping->gp_reg_b,
                                  l_b_offset );
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
                                      i_n_blocking, 0, 1, 0 );

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
                                      i_n_blocking + l_m, 0, 1, 0 );
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
                                                 l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
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
                                                 l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                                                 LIBXSMM_X86_VEC_REG_UNDEF );
          }
        }
      }
    }
  }
}

