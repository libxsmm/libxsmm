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
#include "generator_gemm_avx512_microkernel.h"
#include "generator_x86_instructions.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_nofsdbcst( libxsmm_generated_code*            io_generated_code,
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
      if ((LIBXSMM_GEMM_PRECISION_I8 != LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype)) && (LIBXSMM_GEMM_PRECISION_BF16 != LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype)) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE))) {
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
    /* load column vectors of A upfront */
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->a_vmove_instruction,
          i_gp_reg_mapping->gp_reg_a,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m,
          i_micro_kernel_config->vector_name,
          1+l_m, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, 1, 0 );

      /* current A prefetch, next rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD || i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD ) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT1,
            i_gp_reg_mapping->gp_reg_a,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            ((i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m) + (64 * l_m_blocking) );
      }

      /* prefetch a different A matrix provided by the prefetch pointers */
      if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
            LIBXSMM_X86_INSTR_PREFETCHT1,
            i_gp_reg_mapping->gp_reg_a_prefetch,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (i_micro_kernel_config->datatype_size) * (i_micro_kernel_config->vector_length) * l_m );
      }
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
      if ((l_n < l_m_blocking)  &&  (LIBXSMM_GEMM_PRECISION_I8 != LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype)) && (LIBXSMM_GEMM_PRECISION_BF16 != LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype)) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET) || (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE))) {
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

          /* if we prefetch next A into L2, we need to also increment the prefetch pointer */
          if ( (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2) || (i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                i_micro_kernel_config->alu_add_instruction,
                i_gp_reg_mapping->gp_reg_a_prefetch,
                (i_xgemm_desc->lda)*(i_micro_kernel_config->datatype_size) );
          }
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


LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_fsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                           const unsigned int                 i_n_blocking,
                                                                           const unsigned int                 i_k_blocking )
{
  unsigned int l_n;
  unsigned int l_k;
  unsigned int l_n_accs = 0;

#if !defined(NDEBUG)
  if ( i_n_blocking > 30 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
#endif

  /* compute number of n accumulators to hide FMA latencies */
  if (i_n_blocking >= 12) {
    l_n_accs = 1;
  } else if (i_n_blocking >= 6) {
    l_n_accs = 2;
  } else {
    l_n_accs = 4;
  }
  if ( l_n_accs > i_k_blocking ) {
    l_n_accs = i_k_blocking;
    l_n_accs = (l_n_accs == 0) ? 1 : l_n_accs;
  }

  /* xor additional accumulator, if needed */
  for ( l_k = 1; l_k < l_n_accs; l_k++) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           io_generated_code->arch,
                                           i_micro_kernel_config->vxor_instruction,
                                           i_micro_kernel_config->vector_name,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n );
    }
  }

  /* in case of int8 GEMM on SKX use zmm2 for 16bit 1's */
  if ( (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) &&
         (io_generated_code->arch < LIBXSMM_X86_AVX512_CLX) ) {
    short l_all_ones[32] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                         (const unsigned char *)l_all_ones,
                                                         "my_int16_ones",
                                                         i_micro_kernel_config->vector_name,
                                                         2 );
  }

  /* apply k blocking */
  for ( l_k = 0; l_k < i_k_blocking; l_k++ ) {
    if ( l_k == 0 ) {
       /* load A */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        io_generated_code->arch,
                                        i_micro_kernel_config->a_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size,
                                        i_micro_kernel_config->vector_name,
                                        0,
                                        i_micro_kernel_config->use_masking_a_c, 1, 0 );
      /* current A prefetch, next rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * l_k + i_micro_kernel_config->datatype_size) + 64 );
      }
      if ( i_k_blocking > 1 ) {
        /* second A load in first iteration, in case of large blockings -> hiding L1 latencies */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          io_generated_code->arch,
                                          i_micro_kernel_config->a_vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size,
                                          i_micro_kernel_config->vector_name,
                                          1,
                                          i_micro_kernel_config->use_masking_a_c, 1, 0 );
        /* current A prefetch, next rows for the current column */
        if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD ||
             i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_micro_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_a,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            (i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size) + 64 );
        }
      }
    } else if ( l_k < (i_k_blocking - 1) ) {
      /* pipelined load of A, one k iteration ahead */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        io_generated_code->arch,
                                        i_micro_kernel_config->a_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size,
                                        i_micro_kernel_config->vector_name,
                                        (l_k+1)%2,
                                        i_micro_kernel_config->use_masking_a_c, 1, 0 );
      /* current A prefetch, next rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD          ||
           i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD    ) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_k+1) * i_micro_kernel_config->datatype_size) + 64 );
      }
    }

    /* next A prefetch "same" rows in "same" column, but in a different matrix */
    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ||
         i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C) {
      libxsmm_x86_instruction_prefetch( io_generated_code,
                                        i_micro_kernel_config->prefetch_instruction,
                                        i_gp_reg_mapping->gp_reg_a_prefetch,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (i_xgemm_desc->lda * l_k * i_micro_kernel_config->datatype_size) );
      if ( l_k == (i_k_blocking - 1) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a_prefetch,
                                         i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }
    }

    /* in last k-iteration: advance pointers */
    if ( l_k == (i_k_blocking - 1) ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_a,
                                       i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
    }

    /* in case of bfloat16 "prepare" A matrix in registers zmm l_k%2 and zmm3 using FP32 numbers */
    if ( (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) &&
         (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX)                ) {
      /* we put "0" elements of A matrix into zmm3 */
      libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
          io_generated_code->arch,
          LIBXSMM_X86_INSTR_VPSLLD,
          i_micro_kernel_config->vector_name,
          l_k%2,
          3,
          LIBXSMM_X86_VEC_REG_UNDEF,
          16);

      /* we put "1" elements of A matrix into l_k%2 zmm*/
      libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
          io_generated_code->arch,
          LIBXSMM_X86_INSTR_VPSRAD,
          i_micro_kernel_config->vector_name,
          l_k%2,
          l_k%2,
          LIBXSMM_X86_VEC_REG_UNDEF,
          16);
      libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
          io_generated_code->arch,
          LIBXSMM_X86_INSTR_VPSLLD,
          i_micro_kernel_config->vector_name,
          l_k%2,
          l_k%2,
          LIBXSMM_X86_VEC_REG_UNDEF,
          16);
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++) {
        /* determining base, idx and scale values */
        unsigned int l_b_reg = i_gp_reg_mapping->gp_reg_b;
        unsigned int l_b_idx = LIBXSMM_X86_GP_REG_UNDEF;
        unsigned int l_scale = 0;
        unsigned int l_disp = (l_k*i_micro_kernel_config->datatype_size)+(l_n*i_xgemm_desc->ldb*i_micro_kernel_config->datatype_size);

        if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) || LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                   io_generated_code->arch,
                                                   i_micro_kernel_config->vmul_instruction,
                                                   1,
                                                   l_b_reg,
                                                   l_b_idx,
                                                   l_scale,
                                                   l_disp,
                                                   i_micro_kernel_config->vector_name,
                                                   l_k%2,
                                                   i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
        } else if (LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          if ( io_generated_code->arch == LIBXSMM_X86_AVX512_CORE ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPBROADCASTD,
                                              l_b_reg,
                                              l_b_idx, l_scale,
                                              l_disp,
                                              i_micro_kernel_config->vector_name,
                                              3, 0, 1, 0 );
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPMADDWD,
                                              i_micro_kernel_config->vector_name,
                                              l_k%2,
                                              3,
                                              3 );
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPADDD,
                                              i_micro_kernel_config->vector_name,
                                              3,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CLX ) || ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
            libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                     io_generated_code->arch,
                                                     LIBXSMM_X86_INSTR_VPDPWSSD,
                                                     1,
                                                     l_b_reg,
                                                     l_b_idx,
                                                     l_scale,
                                                     l_disp,
                                                     i_micro_kernel_config->vector_name,
                                                     l_k%2,
                                                     i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          } else {
            /* shouldn't happen */
          }
        } else if (LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          if ( io_generated_code->arch < LIBXSMM_X86_AVX512_CLX ) {
            /* let's broadcast B into zmm3 */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPBROADCASTD,
                                              l_b_reg,
                                              l_b_idx, l_scale,
                                              l_disp,
                                              i_micro_kernel_config->vector_name,
                                              3, 0, 1, 0 );

            /* 8 bit mix-sign Mul */
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0  ) {
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                       io_generated_code->arch,
                                                       LIBXSMM_X86_INSTR_VPMADDUBSW,
                                                       i_micro_kernel_config->vector_name,
                                                       3,
                                                       l_k%2,
                                                       3 );
            } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                       io_generated_code->arch,
                                                       LIBXSMM_X86_INSTR_VPMADDUBSW,
                                                       i_micro_kernel_config->vector_name,
                                                       l_k%2,
                                                       3,
                                                       3 );
            } else {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
              return;
            }

            /* 16 bit mul with 1 */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPMADDWD,
                                              i_micro_kernel_config->vector_name,
                                              2,
                                              3,
                                              3 );

            /* add to accumulator */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VPADDD,
                                              i_micro_kernel_config->vector_name,
                                              3,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          } else if ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
              libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                       io_generated_code->arch,
                                                       LIBXSMM_X86_INSTR_VPDPBUSD,
                                                       1,
                                                       l_b_reg,
                                                       l_b_idx,
                                                       l_scale,
                                                       l_disp,
                                                       i_micro_kernel_config->vector_name,
                                                       l_k%2,
                                                       i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
            } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                                                io_generated_code->arch,
                                                LIBXSMM_X86_INSTR_VPBROADCASTD,
                                                l_b_reg,
                                                l_b_idx, l_scale,
                                                l_disp,
                                                i_micro_kernel_config->vector_name,
                                                3, 0, 1, 0 );

              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                io_generated_code->arch,
                                                LIBXSMM_X86_INSTR_VPDPBUSD,
                                                i_micro_kernel_config->vector_name,
                                                l_k%2,
                                                3,
                                                i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );

            } else {
              LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
              return;
            }
          } else {
            LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
            return;
          }
        } else if (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          if ( io_generated_code->arch < LIBXSMM_X86_AVX512_CPX ) {
            /* broadcast pair of B matrix values into zmm2 */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VBROADCASTSS,
                                              l_b_reg,
                                              l_b_idx, l_scale,
                                              l_disp,
                                              i_micro_kernel_config->vector_name,
                                              2, 0, 1, 0 );

            /* we put "1" elements of B matrix into zmm2 */
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                io_generated_code->arch,
                LIBXSMM_X86_INSTR_VPSRAD,
                i_micro_kernel_config->vector_name,
                2,
                2,
                LIBXSMM_X86_VEC_REG_UNDEF,
                16);
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                io_generated_code->arch,
                LIBXSMM_X86_INSTR_VPSLLD,
                i_micro_kernel_config->vector_name,
                2,
                2,
                LIBXSMM_X86_VEC_REG_UNDEF,
                16);

            /* perform fma operations for multiplying "1" elements of A and B */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VFMADD231PS,
                                              i_micro_kernel_config->vector_name,
                                              l_k%2,
                                              2,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );

            /* broadcast pair of B matrix values into zmm2 */
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VBROADCASTSS,
                                              l_b_reg,
                                              l_b_idx, l_scale,
                                              l_disp,
                                              i_micro_kernel_config->vector_name,
                                              2, 0, 1, 0 );

            /* we put "0" elements of B matrix into zmm2 */
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                io_generated_code->arch,
                LIBXSMM_X86_INSTR_VPSLLD,
                i_micro_kernel_config->vector_name,
                2,
                2,
                LIBXSMM_X86_VEC_REG_UNDEF,
                16);

            /* perform fma operations for multiplying "0" elements of A and B */
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              io_generated_code->arch,
                                              LIBXSMM_X86_INSTR_VFMADD231PS,
                                              i_micro_kernel_config->vector_name,
                                              3,
                                              2,
                                              i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          } else {
            libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                     io_generated_code->arch,
                                                     LIBXSMM_X86_INSTR_VDPBF16PS,
                                                     1,
                                                     l_b_reg,
                                                     l_b_idx,
                                                     l_scale,
                                                     l_disp,
                                                     i_micro_kernel_config->vector_name,
                                                     l_k%2,
                                                     i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
          }
        } else {
          /* should not happen */
        }
      }
    } else {
      for ( l_n = 0; l_n < i_n_blocking; l_n++) {
        libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                 io_generated_code->arch,
                                                 i_micro_kernel_config->vmul_instruction,
                                                 1,
                                                 i_gp_reg_mapping->gp_reg_b,
                                                 LIBXSMM_X86_GP_REG_UNDEF,
                                                 0,
                                                 (l_k*i_xgemm_desc->ldb*i_micro_kernel_config->datatype_size) + (l_n*i_micro_kernel_config->datatype_size),
                                                 i_micro_kernel_config->vector_name,
                                                 l_k%2,
                                                 i_micro_kernel_config->vector_reg_count - (i_n_blocking*((l_k%l_n_accs)+1)) + l_n );
      }
    }
  }

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) == 0 ) {
    /* advance pointers of B only when we are not fully unrolling K and taking care of intermediate advances */
    if ( i_k_blocking < (unsigned int)i_xgemm_desc->k ) {
      /* advance pointers of B */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_b,
                                       i_k_blocking * i_micro_kernel_config->datatype_size );
    }
  } else {
    /* advance pointers of B only when we are not fully unrolling K and taking care of intermediate advances */
    if ( i_k_blocking < (unsigned int)i_xgemm_desc->k ) {
      /* advance B ptr by K rows */
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_b,
                                       (i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->ldb) );
    }
  }

  /* add additional accumulators, if needed */
  for ( l_k = 1; l_k < l_n_accs; l_k++) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      if ( (LIBXSMM_GEMM_PRECISION_F32  == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) ||
           (LIBXSMM_GEMM_PRECISION_F64  == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) ||
           (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ))    ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             io_generated_code->arch,
                                             i_micro_kernel_config->vadd_instruction,
                                             i_micro_kernel_config->vector_name,
                                             i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                             i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n,
                                             i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
      } else if ( (LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) ||
                  (LIBXSMM_GEMM_PRECISION_I8  == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ))    ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             io_generated_code->arch,
                                             LIBXSMM_X86_INSTR_VPADDD,
                                             i_micro_kernel_config->vector_name,
                                             i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                             i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n,
                                             i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
      } else {
        /* shouldn't happen */
      }
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_fsdbcst_qfma( libxsmm_generated_code*            io_generated_code,
                                                                                const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                const unsigned int                 i_n_blocking,
                                                                                const unsigned int                 i_k_blocking )
{
  unsigned int l_n;
  unsigned int l_k;
  unsigned int l_z;
  unsigned int l_n_accs = 0;

#if !defined(NDEBUG)
  if ( i_n_blocking > 28 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }
#endif

  /* lazy fix of when QMADD doesn't work (transB and not FP32/I16)*/
  if ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) != 0) || ( ( LIBXSMM_GEMM_PRECISION_F32 != LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) &&
                                                                     ( LIBXSMM_GEMM_PRECISION_I16 != LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )    ) ) {
    libxsmm_generator_gemm_avx512_microkernel_fsdbcst( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                                                       i_xgemm_desc, i_n_blocking, i_k_blocking );
    return;
  }

  /* compute number of n accumulators to hide FMA latencies */
  if (i_n_blocking >= 14) {
    l_n_accs = 1;
  } else if (i_n_blocking >= 7) {
    l_n_accs = 2;
  } else {
    l_n_accs = 4;
  }
  if ( l_n_accs > (i_k_blocking/4) ) {
    l_n_accs = (i_k_blocking/4);
    l_n_accs = (l_n_accs == 0) ? 1 : l_n_accs;
  }

  /* xor additional accumulator, if needed */
  for ( l_k = 1; l_k < l_n_accs; l_k++) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           io_generated_code->arch,
                                           i_micro_kernel_config->vxor_instruction,
                                           i_micro_kernel_config->vector_name,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n );
    }
  }

  /* apply k blocking */
  for ( l_k = 0; l_k < i_k_blocking; ++l_k ) {
    unsigned int l_lcl_k = (l_k+4 <= i_k_blocking) ? 4 : 1;

    /* load A matrix */
    for ( l_z = 0; l_z < l_lcl_k; l_z++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        io_generated_code->arch,
                                        i_micro_kernel_config->a_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        i_xgemm_desc->lda * (l_k+l_z) * i_micro_kernel_config->datatype_size,
                                        i_micro_kernel_config->vector_name,
                                        l_z,
                                        i_micro_kernel_config->use_masking_a_c, 1, 0 );

      /* current A prefetch, next rows for the current column */
      if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2_AHEAD ||
           i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_k+l_z) * i_micro_kernel_config->datatype_size) + 64 );
      }
    }

    /* next A prefetch "same" rows in "same" column, but in a different matrix */
    if ( i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2          ||
         i_xgemm_desc->prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C    ) {
      for ( l_z = 0; l_z < l_lcl_k; l_z++ ) {
        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          i_micro_kernel_config->prefetch_instruction,
                                          i_gp_reg_mapping->gp_reg_a_prefetch,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_xgemm_desc->lda * (l_k+l_z) * i_micro_kernel_config->datatype_size) );
      }
      if ( (l_k+l_lcl_k) == i_k_blocking ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a_prefetch,
                                         i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
      }
    }

    /* in last k-iteration: advance pointers */
    if ( (l_k+l_lcl_k) == i_k_blocking ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_a,
                                       i_k_blocking * i_micro_kernel_config->datatype_size * i_xgemm_desc->lda );
    }

    /* compute vectorwidth (A) * column broadcast (B) */
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      /* determining base, idx and scale values */
      unsigned int l_b_reg = i_gp_reg_mapping->gp_reg_b;
      unsigned int l_b_idx = LIBXSMM_X86_GP_REG_UNDEF;
      unsigned int l_scale = 0;
      unsigned int l_disp = (l_k*i_micro_kernel_config->datatype_size)+(l_n*i_xgemm_desc->ldb*i_micro_kernel_config->datatype_size);

      if ( l_lcl_k == 4 ) {
        if (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
                                                    io_generated_code->arch,
                                                    LIBXSMM_X86_INSTR_V4FMADDPS,
                                                    l_b_reg,
                                                    l_b_idx,
                                                    l_scale,
                                                    l_disp,
                                                    i_micro_kernel_config->vector_name,
                                                    0,
                                                    i_micro_kernel_config->vector_reg_count - (i_n_blocking*(((l_k/4)%l_n_accs)+1)) + l_n );
        } else if (LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
                                                    io_generated_code->arch,
                                                    LIBXSMM_X86_INSTR_VP4DPWSSD,
                                                    l_b_reg,
                                                    l_b_idx,
                                                    l_scale,
                                                    l_disp,
                                                    i_micro_kernel_config->vector_name,
                                                    0,
                                                    i_micro_kernel_config->vector_reg_count - (i_n_blocking*(((l_k/4)%l_n_accs)+1)) + l_n );
        } else {
          /* shouldn't happen */
        }
      } else {
        if (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                   io_generated_code->arch,
                                                   i_micro_kernel_config->vmul_instruction,
                                                   1,
                                                   l_b_reg,
                                                   l_b_idx,
                                                   l_scale,
                                                   l_disp,
                                                   i_micro_kernel_config->vector_name,
                                                   0,
                                                   i_micro_kernel_config->vector_reg_count - (i_n_blocking*(((l_k)%l_n_accs)+1)) + l_n );
        } else if (LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_K_BLOCK );
          return;
        } else {
          /* shouldn't happen */
        }
      }
    }
    if (l_lcl_k == 4) {
      l_k+=3;
    }
  }

  /* advance pointers of B only when we are not fully unrolling K and taking care of intermediate advances */
  if ( i_k_blocking < (unsigned int)i_xgemm_desc->k ) {
    /* advance pointers of B */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_b,
                                     i_k_blocking * i_micro_kernel_config->datatype_size );
  }

  /* add additional accumulators, if needed */
  for ( l_k = 1; l_k < l_n_accs; l_k++) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           io_generated_code->arch,
                                           i_micro_kernel_config->vadd_instruction,
                                           i_micro_kernel_config->vector_name,
                                           i_micro_kernel_config->vector_reg_count - (i_n_blocking*(l_k+1)) + l_n,
                                           i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n,
                                           i_micro_kernel_config->vector_reg_count - i_n_blocking + l_n );
    }
  }
}
