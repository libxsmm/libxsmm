/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_gemm_avx512_microkernel_nofsdbcst.h"
#include "generator_x86_instructions.h"
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_avx512_microkernel_nofsdbcst( libxsmm_generated_code*             io_generated_code,
                                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                          const unsigned int                 i_m_blocking,
                                                          const unsigned int                 i_n_blocking,
                                                          const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length  == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = i_micro_kernel_config->vector_reg_count - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;

#if !defined(NDEBUG)
  if ( (i_n_blocking > 30) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
  if ( (l_m_blocking < 1) || (l_m_blocking > 4) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_M_BLOCK );
    return;
  }
  if ( (((l_m_blocking*i_n_blocking) + l_m_blocking + 1) > 32) && (i_n_blocking < 7) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_REG_BLOCK );
    return;
  }
#endif

  /* Special case that arises in GEMMS from Resnet50 layers  */
  if (i_n_blocking == 7 && l_m_blocking == 4) {
    if ( i_offset != (-1) ) {
      for ( l_n = 0; l_n < 3; l_n++ ) {
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
      }
      if ( i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1 ) {
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size * i_offset * i_xgemm_desc->ldb);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size * i_offset);
        }
        libxsmm_x86_instruction_prefetch(io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT0,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            l_b_offset + 16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size);
      }
    } else {
      for ( l_n = 0; l_n < 3; l_n++ ) {
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
      }
      if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
        libxsmm_x86_instruction_prefetch(io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT0,
            i_gp_reg_mapping->gp_reg_b,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size);
      }
    }

    /* load column vectors of A and multiply with all broadcasted row entries of B */
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->a_vmove_instruction,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m,
          i_micro_kernel_config->vector_name,
          3, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

      /* In case of batch reduce try to prefetch a few more columns ahead...  */
      if ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE)) {
        unsigned int pf_a_cols_ahead = 16;
        if (i_xgemm_desc->lda == 1024) {
          pf_a_cols_ahead = 4;
        }
        libxsmm_x86_instruction_prefetch( io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT0,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m + pf_a_cols_ahead * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size);
      }

      for ( l_n = 0; l_n < 3; l_n++ ) {
        /* issue fma */
        if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmul_instruction,
                i_micro_kernel_config->vector_name,
                l_n,
                3,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmul_instruction,
                i_micro_kernel_config->vector_name,
                3,
                l_n,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else {
            /* should not happen */
          }
        } else {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmul_instruction,
              i_micro_kernel_config->vector_name,
              3,
              l_n,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
    }

    if ( i_offset != (-1) ) {
      for ( l_n = 3; l_n < 6; l_n++ ) {
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
            l_n-3, 0, 1, 0 );
      }
    } else {
      for ( l_n = 3; l_n < 6; l_n++ ) {
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
            l_n-3, 0, 1, 0 );
      }
    }

    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->a_vmove_instruction,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m,
          i_micro_kernel_config->vector_name,
          3, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
      for ( l_n = 3; l_n < 6; l_n++ ) {
        /* issue fma */
        if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmul_instruction,
                i_micro_kernel_config->vector_name,
                l_n-3,
                3,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmul_instruction,
                i_micro_kernel_config->vector_name,
                3,
                l_n-3,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else {
            /* should not happen */
          }
        } else {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmul_instruction,
              i_micro_kernel_config->vector_name,
              3,
              l_n-3,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
    }

    if ( i_offset != (-1) ) {
      for ( l_n = 6; l_n < 7; l_n++ ) {
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
            l_n-6, 0, 1, 0 );
      }
    } else {
      for ( l_n = 6; l_n < 7; l_n++ ) {
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
            l_n-6, 0, 1, 0 );
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

    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->a_vmove_instruction,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m,
          i_micro_kernel_config->vector_name,
          3, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

      for ( l_n = 6; l_n < 7; l_n++ ) {
        /* post increment early */
        if ( (l_m == (l_m_blocking-1)) && (l_n == 6) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_a,
              (i_xgemm_desc->lda)*(i_micro_kernel_config->datatype_size) );
        }

        /* issue fma */
        if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmul_instruction,
                i_micro_kernel_config->vector_name,
                l_n-6,
                3,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmul_instruction,
                i_micro_kernel_config->vector_name,
                3,
                l_n-6,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else {
            /* should not happen */
          }
        } else {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmul_instruction,
              i_micro_kernel_config->vector_name,
              3,
              l_n-6,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
    }
  } else {
    /* load column vectors of A upront */
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->a_vmove_instruction,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m,
          i_micro_kernel_config->vector_name,
          1+l_m, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
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
            0, 0, 1, 0 );

        if (l_n == i_n_blocking - 1) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
              l_b_offset = (i_micro_kernel_config->datatype_size * i_offset * i_xgemm_desc->ldb);
            } else {
              l_b_offset = (i_micro_kernel_config->datatype_size * i_offset);
            }
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                l_b_offset + 16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size);
          }
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
            0, 0, 1, 0 );

        if (l_n == i_n_blocking - 1) {
          if (i_xgemm_desc->prefetch & LIBXSMM_GEMM_PREFETCH_BL1) {
            libxsmm_x86_instruction_prefetch(io_generated_code,
                LIBXSMM_X86_INSTR_PREFETCHT0,
                i_gp_reg_mapping->gp_reg_b,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                16 * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size);
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
      }

      /* In case of batch reduce try to prefetch a few more columns ahead for A...  */
      if ((l_n < l_m_blocking)  && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE))) {
        unsigned int pf_a_cols_ahead = 16;
        if (i_xgemm_desc->lda == 1024) {
          pf_a_cols_ahead = 4;
        }
        libxsmm_x86_instruction_prefetch( io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT0,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_n + pf_a_cols_ahead * i_xgemm_desc->lda * i_micro_kernel_config->datatype_size);
      }

      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        /* post increment early */
        if ( (l_m == 0) && (l_n == i_n_blocking-1) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_a,
              (i_xgemm_desc->lda)*(i_micro_kernel_config->datatype_size) );
        }
        /* issue fma */
        if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmul_instruction,
                i_micro_kernel_config->vector_name,
                0,
                1+l_m,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmul_instruction,
                i_micro_kernel_config->vector_name,
                1+l_m,
                0,
                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else {
            /* should not happen */
          }
        } else {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmul_instruction,
              i_micro_kernel_config->vector_name,
              1+l_m,
              0,
              l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
    }
  }
}
