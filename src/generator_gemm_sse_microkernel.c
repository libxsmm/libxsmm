/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_gemm_sse_microkernel.h"
#include "generator_common_x86.h"
#include "generator_x86_instructions.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_sse_kloop_kernel( libxsmm_generated_code*            io_generated_code,
                                              const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                              const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                              const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                              const unsigned int                 i_m_blocking,
                                              const unsigned int                 i_n_blocking,
                                              const unsigned int                 i_k_blocking )
{
  unsigned int l_k = 0;
  unsigned int l_k_pack_factor = 1;
  void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*,
                                  const libxsmm_gemm_descriptor*, const unsigned int, const unsigned int, const int);
  unsigned int l_is_i8_uu_ss_gemm = (LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
                                    ( ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) == 0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) == 0) ) ||
                                      ( ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) >  0) && ((i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) >  0) ) );

  /* select correct micro kernel */
  if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
       ((i_xgemm_desc->flags &  LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_sse_microkernel_bf16_vnni_emu;
  } else if ( ( l_is_i8_uu_ss_gemm != 0 ) &&
              ((i_xgemm_desc->flags &  LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_sse_microkernel_int8_uu_ss_vnni_emu;
  } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) &&
              ((i_xgemm_desc->flags &  LIBXSMM_GEMM_FLAG_VNNI_A) == 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_sse_microkernel_bf16_flat_emu;
  } else {
    l_generator_microkernel = libxsmm_generator_gemm_sse_microkernel;
  }

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );;
  }

  for ( l_k = 0; l_k < i_k_blocking; l_k += l_k_pack_factor) {
    l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                            i_xgemm_desc, i_m_blocking, i_n_blocking,
                            ( i_k_blocking == (unsigned int)i_xgemm_desc->k ) ? (int)l_k : -1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_sse_microkernel( libxsmm_generated_code*            io_generated_code,
                                             const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                             const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                             const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                             const unsigned int                 i_m_blocking,
                                             const unsigned int                 i_n_blocking,
                                             const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;

  /* check that m_blocking is a multiple of vlen and that n_blocking is valid */
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_SSE_AVX2_LP_HELPER_PTR, i_gp_reg_mapping->gp_reg_help_1 );
  }

  if (l_m_blocking == 1) {
    /* load column vectors of A */
    libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                  i_micro_kernel_config->a_vmove_instruction,
                                  i_gp_reg_mapping->gp_reg_a,
                                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                                  0,
                                  i_micro_kernel_config->vector_name,
                                  i_n_blocking, i_micro_kernel_config->use_masking_a_c, i_m_blocking%i_micro_kernel_config->vector_length, 0 );
    /* loop over columns of B */
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* post increment of a pointer early */
      if ( l_n == 0 ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
      }
      /* different ways of using B */
      if ( i_offset != (-1) ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in);
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        /* generate shuffle as SSE has no broadcast load for single precision, SSE2 has no broadcast at all */
        if ( ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ||
               LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ||
               LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) || (io_generated_code->arch == LIBXSMM_X86_GENERIC))
             && ( i_micro_kernel_config->b_shuff_instruction != LIBXSMM_X86_INSTR_UNDEF ) ) {
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, i_micro_kernel_config->b_shuff_instruction,
                                                         i_micro_kernel_config->vector_name, l_n, l_n, 0 );
        }
      } else {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        /* generate shuffle as SSE has no broadcast load for single precision, SSE2 has no broadcast at all */
        if ( ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ||
               LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ||
               LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) || (io_generated_code->arch == LIBXSMM_X86_GENERIC))
             && ( i_micro_kernel_config->b_shuff_instruction != LIBXSMM_X86_INSTR_UNDEF ) ) {
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, i_micro_kernel_config->b_shuff_instruction,
                                                         i_micro_kernel_config->vector_name, l_n, l_n, 0 );
        }
        if ( l_n == (i_n_blocking -1) ) {
          /* handle trans B */
          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
            l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
          } else {
            l_b_offset = i_micro_kernel_config->datatype_size_in*l_k_pack_factor;
          }

          libxsmm_x86_instruction_alu_imm( io_generated_code,
                                       i_micro_kernel_config->alu_add_instruction,
                                       i_gp_reg_mapping->gp_reg_b,
                                       l_b_offset );
        }
      }
      if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
        unsigned int l_u_reg = 0;
        unsigned int l_s_reg = 0;

        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
          l_u_reg = l_n;
          l_s_reg = i_n_blocking+1;
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_MOVUPS,
                                                    i_micro_kernel_config->vector_name,
                                                    i_n_blocking,
                                                    l_s_reg );
        } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
          l_u_reg = i_n_blocking;
          l_s_reg = l_n;
        } else {
          /* should not happen */
        }

        /* issue mul-add */
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                  i_micro_kernel_config->vmul_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  l_u_reg,
                                                  l_s_reg );
        libxsmm_x86_instruction_vec_compute_mem_1reg( io_generated_code,
                                                      LIBXSMM_X86_INSTR_PMADDWD,
                                                      i_micro_kernel_config->vector_name,
                                                      i_gp_reg_mapping->gp_reg_help_1,
                                                      LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 0,
                                                      l_s_reg );
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                  i_micro_kernel_config->vadd_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  l_s_reg,
                                                  l_vec_reg_acc_start + l_n );
      } else {
        /* issue mul-add */
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                  i_micro_kernel_config->vmul_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  i_n_blocking,
                                                  l_n );
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                  i_micro_kernel_config->vadd_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  l_n,
                                                  l_vec_reg_acc_start + l_n );

      }
    }
  } else {
    /* broadcast from B -> into vec registers 0 to i_n_blocking */
    if ( i_offset != (-1) ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in);
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        /* generate shuffle as SSE has no broadcast load for single precision, SSE2 has no broadcast at all */
        if ( ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ||
               LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ||
               LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) || (io_generated_code->arch == LIBXSMM_X86_GENERIC))
             && ( i_micro_kernel_config->b_shuff_instruction != LIBXSMM_X86_INSTR_UNDEF ) ) {
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, i_micro_kernel_config->b_shuff_instruction,
                                                         i_micro_kernel_config->vector_name, l_n, l_n, 0 );
        }
      }
    } else {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        /* generate shuffle as SSE has no broadcast load for single precision, SSE2 has no broadcast at all */
        if ( ((LIBXSMM_DATATYPE_F32 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ||
               LIBXSMM_DATATYPE_I8  == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ||
               LIBXSMM_DATATYPE_I16 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype )) || (io_generated_code->arch == LIBXSMM_X86_GENERIC))
             && ( i_micro_kernel_config->b_shuff_instruction != LIBXSMM_X86_INSTR_UNDEF ) ) {
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, i_micro_kernel_config->b_shuff_instruction,
                                                         i_micro_kernel_config->vector_name, l_n, l_n, 0 );
        }
      }
      /* handle trans B */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
      } else {
        l_b_offset = i_micro_kernel_config->datatype_size_in*l_k_pack_factor;
      }

      libxsmm_x86_instruction_alu_imm( io_generated_code,
                                  i_micro_kernel_config->alu_add_instruction,
                                  i_gp_reg_mapping->gp_reg_b,
                                  l_b_offset );
    }

    /* load column vectors of A and multiply with all broadcasted row entries of B */
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
          unsigned int l_u_reg = 0;
          unsigned int l_s_reg = 0;

          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                              i_micro_kernel_config->a_vmove_instruction,
                                              i_gp_reg_mapping->gp_reg_a,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                              i_micro_kernel_config->vector_name,
                                              i_n_blocking, (l_m == (l_m_blocking-1)) ? i_micro_kernel_config->use_masking_a_c : 0, i_m_blocking%i_micro_kernel_config->vector_length, 0 );

          if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
            l_u_reg = l_n;
            l_s_reg = i_n_blocking;
          } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
            l_u_reg = i_n_blocking;
            l_s_reg = i_n_blocking + 1;
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                      LIBXSMM_X86_INSTR_MOVUPS,
                                                      i_micro_kernel_config->vector_name,
                                                      l_n,
                                                      l_s_reg );
          } else {
            /* should not happen */
          }

          /* issue mul-add */
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                    i_micro_kernel_config->vmul_instruction,
                                                    i_micro_kernel_config->vector_name,
                                                    l_u_reg,
                                                    l_s_reg );
          libxsmm_x86_instruction_vec_compute_mem_1reg( io_generated_code,
                                                        LIBXSMM_X86_INSTR_PMADDWD,
                                                        i_micro_kernel_config->vector_name,
                                                        i_gp_reg_mapping->gp_reg_help_1,
                                                        LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 0,
                                                        l_s_reg );
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                    i_micro_kernel_config->vadd_instruction,
                                                    i_micro_kernel_config->vector_name,
                                                    l_s_reg,
                                                    l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        } else {
          if ( (l_m < (l_m_blocking-1)) || (l_n == 0) ) {
            libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                              i_micro_kernel_config->a_vmove_instruction,
                                              i_gp_reg_mapping->gp_reg_a,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                              i_micro_kernel_config->vector_name,
                                              i_n_blocking, (l_m == (l_m_blocking-1)) ? i_micro_kernel_config->use_masking_a_c : 0, i_m_blocking%i_micro_kernel_config->vector_length, 0 );
          }

          if ( l_m < (l_m_blocking-1) ) {
            /* issue mul+add */
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                      i_micro_kernel_config->vmul_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      l_n,
                                                      i_n_blocking );
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                      i_micro_kernel_config->vadd_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      i_n_blocking,
                                                      l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          } else {
            /* issue mul+add */
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                      i_micro_kernel_config->vmul_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      i_n_blocking,
                                                      l_n );
            libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                      i_micro_kernel_config->vadd_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      l_n,
                                                      l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          }
        }
      }
    }
    /* increae a pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_sse_microkernel_int8_uu_ss_vnni_emu( libxsmm_generated_code*            io_generated_code,
                                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                 const unsigned int                 i_m_blocking,
                                                                 const unsigned int                 i_n_blocking,
                                                                 const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* we process bf16 vnni in two passes and match AVX512 numerics */
  unsigned int l_pass = 0;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;

  /* check that m_blocking is a multiple of vlen and that n_blocking is valid */
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  for ( l_pass = 0; l_pass < 2; ++l_pass ) {
    /* broadcast from B -> into vec registers 0 to i_n_blocking */
    if ( i_offset != (-1) ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in);
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, i_micro_kernel_config->b_shuff_instruction,
                                                       i_micro_kernel_config->vector_name, l_n, l_n, 0 );
      }
    } else {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in * l_k_pack_factor;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, i_micro_kernel_config->b_shuff_instruction,
                                                       i_micro_kernel_config->vector_name, l_n, l_n, 0 );
      }
      if ( l_pass == 1 ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
        } else {
          l_b_offset = i_micro_kernel_config->datatype_size_in * l_k_pack_factor;
        }

        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                      i_micro_kernel_config->alu_add_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      l_b_offset );
      }
    }

    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* let's create the tuples of int8 values */
      if ( l_pass == 0 ) {
        libxsmm_x86_instruction_vec_compute_1reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PSLLD_I,
                                                       i_micro_kernel_config->vector_name,
                                                       l_n, 16 );
        libxsmm_x86_instruction_vec_compute_1reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PSRLD_I,
                                                       i_micro_kernel_config->vector_name,
                                                       l_n, 16 );
      } else {
        libxsmm_x86_instruction_vec_compute_1reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PSRLD_I,
                                                       i_micro_kernel_config->vector_name,
                                                       l_n, 16 );
      }

      /* convert 32bit int to 16bit int */
      libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_PACKUSDW,
                                               i_micro_kernel_config->vector_name, l_n, l_n);

      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_PMOVZXBW,
                                                 i_micro_kernel_config->vector_name, l_n, l_n);
      } else {
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_PMOVSXBW,
                                                 i_micro_kernel_config->vector_name, l_n, l_n);
      }
    }

     /* load column vectors of A and multiply with all broadcasted row entries of B */
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                     i_micro_kernel_config->a_vmove_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     LIBXSMM_X86_GP_REG_UNDEF, 0,
                                     (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                     i_micro_kernel_config->vector_name,
                                     i_n_blocking, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, i_m_blocking%i_micro_kernel_config->vector_length, 0 );

        /* create tuples of int8 values */
        if ( l_pass == 0 ) {
          libxsmm_x86_instruction_vec_compute_1reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PSLLD_I,
                                                         i_micro_kernel_config->vector_name,
                                                         i_n_blocking, 16 );
          libxsmm_x86_instruction_vec_compute_1reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PSRLD_I,
                                                         i_micro_kernel_config->vector_name,
                                                         i_n_blocking, 16 );
        } else {
          libxsmm_x86_instruction_vec_compute_1reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PSRLD_I,
                                                         i_micro_kernel_config->vector_name,
                                                         i_n_blocking, 16 );
        }

        /* convert 32bit int to 16bit int */
        libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_PACKUSDW,
                                                 i_micro_kernel_config->vector_name, i_n_blocking, i_n_blocking);

        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
          libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_PMOVZXBW,
                                                   i_micro_kernel_config->vector_name, i_n_blocking, i_n_blocking);
        } else {
          libxsmm_x86_instruction_vec_compute_2reg(io_generated_code, LIBXSMM_X86_INSTR_PMOVSXBW,
                                                   i_micro_kernel_config->vector_name, i_n_blocking, i_n_blocking);
        }

        /* post increment early */
        if ( (l_m == (l_m_blocking-1)) && (l_n == (i_n_blocking-1)) && (l_pass == 1) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
                                           i_micro_kernel_config->alu_add_instruction,
                                           i_gp_reg_mapping->gp_reg_a,
                                           (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
        }

        /* issue fma */
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                  LIBXSMM_X86_INSTR_PMADDWD,
                                                  i_micro_kernel_config->vector_name,
                                                  l_n,
                                                  i_n_blocking );
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                  LIBXSMM_X86_INSTR_PADDD,
                                                  i_micro_kernel_config->vector_name,
                                                  i_n_blocking,
                                                  l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_sse_microkernel_bf16_vnni_emu( libxsmm_generated_code*            io_generated_code,
                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                           const unsigned int                 i_m_blocking,
                                                           const unsigned int                 i_n_blocking,
                                                           const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
  /* register blocking counter in n */
  unsigned int l_n = 0;
  /* register blocking counter in m */
  unsigned int l_m = 0;
  /* start register of accumulator */
  unsigned int l_vec_reg_acc_start = 16 - (i_n_blocking * l_m_blocking);
  /* temp variable for b-offset to handle no-trans/trans B */
  int l_b_offset = 0;
  /* we process bf16 vnni in two passes and match AVX512 numerics */
  unsigned int l_pass = 0;
  /* k packing factor for VNNI */
  unsigned int l_k_pack_factor = 1;

  /* check that m_blocking is a multiple of vlen and that n_blocking is valid */
  if ( (i_n_blocking > 3) || (i_n_blocking < 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype) );
  }

  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_SSE_AVX2_LP_HELPER_PTR, i_gp_reg_mapping->gp_reg_help_1 );

  for ( l_pass = 0; l_pass < 2; ++l_pass ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_MOVUPS,
                                      i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'x', i_n_blocking, 0, 0, 0 );
    /* broadcast from B -> into vec registers 0 to i_n_blocking */
    if ( i_offset != (-1) ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in);
        } else {
          l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in);
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, i_micro_kernel_config->b_shuff_instruction,
                                                       i_micro_kernel_config->vector_name, l_n, l_n, 0 );


        /* in the firs pass we conver the higher BF16 values to FP32; in the second the lower Bf16 values */
        if ( l_pass == 0 ) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_PAND,
                                                         i_micro_kernel_config->vector_name,
                                                         i_n_blocking, l_n );

        } else {
          libxsmm_x86_instruction_vec_compute_1reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PSLLD_I,
                                                         i_micro_kernel_config->vector_name,
                                                         l_n, 16 );
        }
      }
    } else {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = l_n * i_micro_kernel_config->datatype_size_in * l_k_pack_factor;
        } else {
          l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in;
        }

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->b_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, i_micro_kernel_config->b_shuff_instruction,
                                                       i_micro_kernel_config->vector_name, l_n, l_n, 0 );

        /* in the firs pass we conver the higher BF16 values to FP32; in the second the lower Bf16 values */
        if ( l_pass == 0 ) {
          libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_PAND,
                                                         i_micro_kernel_config->vector_name,
                                                         i_n_blocking, l_n );

        } else {
          libxsmm_x86_instruction_vec_compute_1reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PSLLD_I,
                                                         i_micro_kernel_config->vector_name,
                                                         l_n, 16 );
        }
      }
      if ( l_pass == 1 ) {
        /* handle trans B */
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
          l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
        } else {
          l_b_offset = i_micro_kernel_config->datatype_size_in * l_k_pack_factor;
        }

        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                      i_micro_kernel_config->alu_add_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      l_b_offset );
      }
    }

    /* load column vectors of A and multiply with all broadcasted row entries of B */
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                     i_micro_kernel_config->a_vmove_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     LIBXSMM_X86_GP_REG_UNDEF, 0,
                                     (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                     i_micro_kernel_config->vector_name,
                                     i_n_blocking, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, i_m_blocking%i_micro_kernel_config->vector_length, 0 );

        /* in the firs pass we conver the higher BF16 values to FP32; in the second the lower Bf16 values */
        if ( l_pass == 0 ) {
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PBLENDW,
                                                         i_micro_kernel_config->vector_name,
                                                         l_n, i_n_blocking, 0x55 );
        } else {
          libxsmm_x86_instruction_vec_compute_1reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PSLLD_I,
                                                         i_micro_kernel_config->vector_name,
                                                         i_n_blocking, 16 );
        }

        /* post increment early */
        if ( (l_m == (l_m_blocking-1)) && (l_n == (i_n_blocking-1)) && (l_pass == 1) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
                                           i_micro_kernel_config->alu_add_instruction,
                                           i_gp_reg_mapping->gp_reg_a,
                                           (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
        }

        /* issue fma */
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                  i_micro_kernel_config->vmul_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  l_n,
                                                  i_n_blocking );
        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                  i_micro_kernel_config->vadd_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  i_n_blocking,
                                                  l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_sse_microkernel_bf16_flat_emu( libxsmm_generated_code*            io_generated_code,
                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                           const unsigned int                 i_m_blocking,
                                                           const unsigned int                 i_n_blocking,
                                                           const int                          i_offset )
{
  /* deriving register blocking from kernel config */
  unsigned int l_m_blocking = ( i_m_blocking % i_micro_kernel_config->vector_length == 0 ) ? i_m_blocking/i_micro_kernel_config->vector_length : (i_m_blocking/i_micro_kernel_config->vector_length)+1;
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

  /* broadcast from B -> into vec registers 0 to i_n_blocking */
  if ( i_offset != (-1) ) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* handle trans B */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset * i_xgemm_desc->ldb) + (l_n * i_micro_kernel_config->datatype_size_in);
      } else {
        l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in);
      }

      libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVW, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 0, l_b_offset,
                                       i_gp_reg_mapping->gp_reg_help_1, 0);
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mapping->gp_reg_help_1, 16 );
      libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PINSRD, 'x', i_gp_reg_mapping->gp_reg_help_1, l_n, 0x0 );
      libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, i_micro_kernel_config->b_shuff_instruction,
                                                     i_micro_kernel_config->vector_name, l_n, l_n, 0 );
    }
  } else {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      /* handle trans B */
      if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
        l_b_offset = l_n * i_micro_kernel_config->datatype_size_in;
      } else {
        l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in;
      }

      libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVW, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 0, l_b_offset,
                                       i_gp_reg_mapping->gp_reg_help_1, 0);
      libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SHLQ, i_gp_reg_mapping->gp_reg_help_1, 16 );
      libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_PINSRD, 'x', i_gp_reg_mapping->gp_reg_help_1, l_n, 0x0 );
      libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, i_micro_kernel_config->b_shuff_instruction,
                                                     i_micro_kernel_config->vector_name, l_n, l_n, 0 );
    }
    /* handle trans B */
    if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      l_b_offset = i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in;
    } else {
      l_b_offset = i_micro_kernel_config->datatype_size_in;
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_b,
                                     l_b_offset );
  }

  /* load column vectors of A and multiply with all broadcasted row entries of B */
  for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                   (( l_m == (l_m_blocking - 1) ) && (i_micro_kernel_config->use_masking_a_c != 0 )) ? LIBXSMM_X86_INSTR_VMOVDQU16 : LIBXSMM_X86_INSTR_MOVSD/* @TODO this is a hack to signal 16bit element load */,
                                   i_gp_reg_mapping->gp_reg_a,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0,
                                   (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m,
                                   i_micro_kernel_config->vector_name,
                                   i_n_blocking, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, i_m_blocking%i_micro_kernel_config->vector_length, 0 );

      /* use regular helper to up convert */
      libxsmm_generator_cvtbf16ps_sse_avx2_avx512( io_generated_code, 'x',
                                                   i_n_blocking, i_n_blocking );

      /* post increment early */
      if ( (l_m == (l_m_blocking-1)) && (l_n == (i_n_blocking-1)) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a,
                                         (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in );
      }

      /* issue fma */
      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                i_micro_kernel_config->vmul_instruction,
                                                i_micro_kernel_config->vector_name,
                                                l_n,
                                                i_n_blocking );
      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                                i_micro_kernel_config->vadd_instruction,
                                                i_micro_kernel_config->vector_name,
                                                i_n_blocking,
                                                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
    }
  }
}

