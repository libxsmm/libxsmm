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
#include "generator_gemm_avx2_microkernel.h"
#include "generator_common_x86.h"
#include "generator_x86_instructions.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_avx2_kloop_kernel( libxsmm_generated_code*            io_generated_code,
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

  /* select correct micro kernel */
  if ( ( (LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) || (LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) ) &&
       (io_generated_code->arch < LIBXSMM_X86_AVX2_ADL) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx2_microkernel_int8_int16_vnni_emu;
  } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) &&
              ((i_xgemm_desc->flags &  LIBXSMM_GEMM_FLAG_VNNI_A) > 0) &&
              (io_generated_code->arch < LIBXSMM_X86_AVX2_SRF) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx2_microkernel_bf16_vnni_emu;
  } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) &&
              ((i_xgemm_desc->flags &  LIBXSMM_GEMM_FLAG_VNNI_A) == 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx2_microkernel_bf16_flat_emu;
  } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )) &&
              ((i_xgemm_desc->flags &  LIBXSMM_GEMM_FLAG_VNNI_A) > 0) ) {
    l_generator_microkernel = libxsmm_generator_gemm_avx2_microkernel_bf16_vnni_srf;
  } else {
    l_generator_microkernel = libxsmm_generator_gemm_avx2_microkernel;
  }

  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype) );;
  }

  for ( l_k = 0; l_k < i_k_blocking; l_k += l_k_pack_factor) {
    l_generator_microkernel(io_generated_code, i_gp_reg_mapping, i_micro_kernel_config,
                            i_xgemm_desc, i_m_blocking, i_n_blocking,
                            ( i_k_blocking == (unsigned int)i_xgemm_desc->k ) ? (int)l_k : -1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_avx2_microkernel( libxsmm_generated_code*            io_generated_code,
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

  if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype) );
  }

  if (l_m_blocking == 1) {
    if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                        i_gp_reg_mapping->gp_reg_help_0, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', i_n_blocking, 0, 0, 0 );
    }

    /* load column vectors of A */
    libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                  i_micro_kernel_config->a_vmove_instruction,
                                  i_gp_reg_mapping->gp_reg_a,
                                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                                  0,
                                  i_micro_kernel_config->vector_name,
                                  i_n_blocking, i_micro_kernel_config->use_masking_a_c, i_n_blocking, 0 );
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
      } else {
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
        if ( l_n == (i_n_blocking -1) ) {
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
      /* issue fma */
      if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VPDPBUSD,
                                                    i_micro_kernel_config->vector_name,
                                                    l_n,
                                                    i_n_blocking,
                                                    l_vec_reg_acc_start + l_n );
        } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VPDPBUSD,
                                                    i_micro_kernel_config->vector_name,
                                                    i_n_blocking,
                                                    l_n,
                                                    l_vec_reg_acc_start + l_n );
        } else {
          /* should not happen */
        }
      } else {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  i_micro_kernel_config->vmul_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  i_n_blocking,
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
      }
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

    if (l_m_blocking == 4) {
      /* load column vectors of A and multiply with all broadcasted row entries of B */
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        if ( ( l_m == (l_m_blocking - 1) ) && (i_micro_kernel_config->use_masking_a_c != 0) ) {
          libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                            i_gp_reg_mapping->gp_reg_help_0, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', i_n_blocking, 0, 0, 0 );
        }

        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                      i_micro_kernel_config->a_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_a,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                      i_micro_kernel_config->vector_name,
                                      i_n_blocking, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, i_n_blocking, 0 );

        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          /* post increment early */
          if ( (l_m == (l_m_blocking-1)) && (l_n == 0) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a,
                                         (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in * l_k_pack_factor);
          }
          /* issue fma */
          if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                        LIBXSMM_X86_INSTR_VPDPBUSD,
                                                        i_micro_kernel_config->vector_name,
                                                        l_n,
                                                        i_n_blocking,
                                                        l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
            } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                        LIBXSMM_X86_INSTR_VPDPBUSD,
                                                        i_micro_kernel_config->vector_name,
                                                        i_n_blocking,
                                                        l_n,
                                                        l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
            } else {
              /* should not happen */
            }
          } else {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                      i_micro_kernel_config->vmul_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      i_n_blocking,
                                                      l_n,
                                                      l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          }
        }
      }
    } else {
      /* load column vectors of A and multiply with all broadcasted row entries of B */
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        if ( ( l_m == (l_m_blocking - 1) ) && (i_micro_kernel_config->use_masking_a_c != 0) ) {
          libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                            i_gp_reg_mapping->gp_reg_help_0, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', i_n_blocking+l_m, 0, 0, 0 );
        }

        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                      i_micro_kernel_config->a_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_a,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                      i_micro_kernel_config->vector_name,
                                      i_n_blocking+l_m, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, i_n_blocking+l_m, 0 );
      }
      for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          /* post increment early */
          if ( (l_m == (l_m_blocking-1)) && (l_n == 0) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a,
                                         (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
          }
          /* issue fma */
          if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
            if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                        LIBXSMM_X86_INSTR_VPDPBUSD,
                                                        i_micro_kernel_config->vector_name,
                                                        l_n,
                                                        i_n_blocking+l_m,
                                                        l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
            } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
              libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                        LIBXSMM_X86_INSTR_VPDPBUSD,
                                                        i_micro_kernel_config->vector_name,
                                                        i_n_blocking+l_m,
                                                        l_n,
                                                        l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
            } else {
              /* should not happen */
            }
          } else {
            libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                      i_micro_kernel_config->vmul_instruction,
                                                      i_micro_kernel_config->vector_name,
                                                      i_n_blocking+l_m,
                                                      l_n,
                                                      l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
          }
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_avx2_microkernel_int8_int16_vnni_emu( libxsmm_generated_code*            io_generated_code,
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
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype) );
  }

  if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_LP_HELPER_PTR, i_gp_reg_mapping->gp_reg_help_1 );

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
    }
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

  /* load column vectors of A and multiply with all broadcasted row entries of B */
  for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
    for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
      if ( ( l_m == (l_m_blocking - 1) ) && (i_micro_kernel_config->use_masking_a_c != 0) ) {
        libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                          i_gp_reg_mapping->gp_reg_help_0, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', i_n_blocking, 0, 0, 0 );
      }

      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                   i_micro_kernel_config->a_vmove_instruction,
                                   i_gp_reg_mapping->gp_reg_a,
                                   LIBXSMM_X86_GP_REG_UNDEF, 0,
                                   (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                   i_micro_kernel_config->vector_name,
                                   i_n_blocking, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, i_n_blocking, 0 );

      /* post increment early */
      if ( (l_m == (l_m_blocking-1)) && (l_n == (i_n_blocking-1)) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a,
                                         (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
      }

      /* issue fma */
      if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
        if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_A_UNSIGNED) > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VPMADDUBSW,
                                                    i_micro_kernel_config->vector_name,
                                                    l_n,
                                                    i_n_blocking,
                                                    i_n_blocking );
        } else if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_B_UNSIGNED) > 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    LIBXSMM_X86_INSTR_VPMADDUBSW,
                                                    i_micro_kernel_config->vector_name,
                                                    i_n_blocking,
                                                    l_n,
                                                    i_n_blocking );
        } else {
          /* should not happen */
        }

        libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code,
                                                      LIBXSMM_X86_INSTR_VPMADDWD,
                                                      i_micro_kernel_config->vector_name,
                                                      i_gp_reg_mapping->gp_reg_help_1,
                                                      LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 0,
                                                      i_n_blocking, i_n_blocking );
      } else if ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  LIBXSMM_X86_INSTR_VPMADDWD,
                                                  i_micro_kernel_config->vector_name,
                                                  i_n_blocking,
                                                  l_n,
                                                  i_n_blocking );
      }
      /* add to accumulator */
      libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                LIBXSMM_X86_INSTR_VPADDD,
                                                i_micro_kernel_config->vector_name,
                                                i_n_blocking,
                                                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n),
                                                l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_avx2_microkernel_bf16_vnni_emu( libxsmm_generated_code*            io_generated_code,
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
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype) );
  }

  if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_LP_HELPER_PTR, i_gp_reg_mapping->gp_reg_help_1 );

  for ( l_pass = 0; l_pass < 2; ++l_pass ) {
    libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                      i_gp_reg_mapping->gp_reg_help_1, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', i_n_blocking, 0, 0, 0 );
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

        /* in the firs pass we conver the higher BF16 values to FP32; in the second the lower Bf16 values */
        if ( l_pass == 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPANDD,
                                                         i_micro_kernel_config->vector_name,
                                                         l_n, i_n_blocking, l_n );

        } else {
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                         i_micro_kernel_config->vector_name,
                                                         l_n, l_n, 16 );
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

        /* in the firs pass we conver the higher BF16 values to FP32; in the second the lower Bf16 values */
        if ( l_pass == 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPANDD,
                                                         i_micro_kernel_config->vector_name,
                                                         l_n, i_n_blocking, l_n );

        } else {
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                         i_micro_kernel_config->vector_name,
                                                         l_n, l_n, 16 );
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
      if ( (i_micro_kernel_config->use_masking_a_c == 0) || ((i_micro_kernel_config->use_masking_a_c == 1) && (l_m < (l_m_blocking-1))) ) {
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                     i_micro_kernel_config->a_vmove_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     LIBXSMM_X86_GP_REG_UNDEF, 0,
                                     (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                     i_micro_kernel_config->vector_name,
                                     i_n_blocking, 0, 0, 0 );

        if ( l_pass == 0 ) {
          libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPBLENDW,
                                                         i_micro_kernel_config->vector_name,
                                                         i_n_blocking, 0, i_n_blocking, 0xaa );
        } else {
          libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                         i_micro_kernel_config->vector_name,
                                                         i_n_blocking, i_n_blocking, 16 );
        }

        /* post increment early */
        if ( (l_m == (l_m_blocking-1)) && (l_pass == 1) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
                                           i_micro_kernel_config->alu_add_instruction,
                                           i_gp_reg_mapping->gp_reg_a,
                                           (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
        }

        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          /* issue fma */
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    i_micro_kernel_config->vmul_instruction,
                                                    i_micro_kernel_config->vector_name,
                                                    i_n_blocking,
                                                    l_n,
                                                    l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      } else {
        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                            i_gp_reg_mapping->gp_reg_help_0, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', i_n_blocking, 0, 0, 0 );

          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                       i_micro_kernel_config->a_vmove_instruction,
                                       i_gp_reg_mapping->gp_reg_a,
                                       LIBXSMM_X86_GP_REG_UNDEF, 0,
                                       (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                       i_micro_kernel_config->vector_name,
                                       i_n_blocking, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, i_n_blocking, 0 );

          /* in the firs pass we conver the higher BF16 values to FP32; in the second the lower Bf16 values */
          if ( l_pass == 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPBLENDW,
                                                           i_micro_kernel_config->vector_name,
                                                           i_n_blocking, l_n, i_n_blocking, 0xaa );
          } else {
            libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                           i_micro_kernel_config->vector_name,
                                                           i_n_blocking, i_n_blocking, 16 );
          }

          /* post increment early */
          if ( (l_m == (l_m_blocking-1)) && (l_n == (i_n_blocking-1)) && (l_pass == 1) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                                             i_micro_kernel_config->alu_add_instruction,
                                             i_gp_reg_mapping->gp_reg_a,
                                             (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
          }

          /* issue fma */
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    i_micro_kernel_config->vmul_instruction,
                                                    i_micro_kernel_config->vector_name,
                                                    i_n_blocking,
                                                    l_n,
                                                    l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_avx2_microkernel_bf16_flat_emu( libxsmm_generated_code*            io_generated_code,
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

  /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_K_BLOCK );
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

      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->b_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_b,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_b_offset,
                                        i_micro_kernel_config->vector_name,
                                        l_n, 0, 1, 0 );

      libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                     i_micro_kernel_config->vector_name,
                                                     l_n, l_n, 16 );
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

      libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                     i_micro_kernel_config->vector_name,
                                                     l_n, l_n, 16 );
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
    if ( (i_micro_kernel_config->use_masking_a_c == 0) || ((i_micro_kernel_config->use_masking_a_c == 1) && (l_m < (l_m_blocking-1))) ) {
      libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                     i_micro_kernel_config->a_vmove_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     LIBXSMM_X86_GP_REG_UNDEF, 0,
                                     (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m,
                                     'x',
                                     i_n_blocking, 0, 0, 0 );

      libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXWD,
                                                i_micro_kernel_config->vector_name,
                                                i_n_blocking, i_n_blocking );

      libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                     i_micro_kernel_config->vector_name,
                                                     i_n_blocking, i_n_blocking, 16 );
       /* post increment early */
      if ( l_m == (l_m_blocking-1) ) {
        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                         i_micro_kernel_config->alu_add_instruction,
                                         i_gp_reg_mapping->gp_reg_a,
                                         (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in );
      }

      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        /* issue fma */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  i_micro_kernel_config->vmul_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  i_n_blocking,
                                                  l_n,
                                                  l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
    } else {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        libxsmm_generator_maskedload_16bit_avx2( io_generated_code, i_gp_reg_mapping->gp_reg_help_0,
                                                 i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                 (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m,
                                                 i_n_blocking, i_m_blocking % i_micro_kernel_config->vector_length );

        libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXWD,
                                                  i_micro_kernel_config->vector_name,
                                                  i_n_blocking, i_n_blocking );

        libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                       i_micro_kernel_config->vector_name,
                                                       i_n_blocking, i_n_blocking, 16 );

        /* post increment early */
        if ( (l_m == (l_m_blocking-1)) && (l_n == (i_n_blocking-1)) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
                                           i_micro_kernel_config->alu_add_instruction,
                                           i_gp_reg_mapping->gp_reg_a,
                                           (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in );
        }

        /* issue fma */
        libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                  i_micro_kernel_config->vmul_instruction,
                                                  i_micro_kernel_config->vector_name,
                                                  i_n_blocking,
                                                  l_n,
                                                  l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_avx2_microkernel_bf16_vnni_srf( libxsmm_generated_code*            io_generated_code,
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

  /* we don't handle Btrans for BF16 */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
    return;
  }

    /* for VNNI we are stepping through to pack ks */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_VNNI_A) == LIBXSMM_GEMM_FLAG_VNNI_A ) {
    l_k_pack_factor = libxsmm_cpuid_dot_pack_factor( (libxsmm_datatype)LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype) );
  }

  if ( i_micro_kernel_config->use_masking_a_c != 0 ) {
    libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_MASK_PTR, i_gp_reg_mapping->gp_reg_help_0 );
  }

  libxsmm_generator_gemm_getval_stack_var( io_generated_code, i_micro_kernel_config, LIBXSMM_GEMM_STACK_VAR_AVX2_LP_HELPER_PTR, i_gp_reg_mapping->gp_reg_help_1 );

  for ( l_pass = 0; l_pass < 2; ++l_pass ) {
    /* broadcast from B -> into vec registers 0 to i_n_blocking */
    if ( i_offset != (-1) ) {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        l_b_offset = (i_micro_kernel_config->datatype_size_in * i_offset) + (i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in);
        l_b_offset += (l_pass == 0) ? 2 : 0; /* we are emulating AVX512 VDPBF16PS */

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      LIBXSMM_X86_INSTR_VBCSTNEBF162PS,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
      }
    } else {
      for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
        l_b_offset = i_xgemm_desc->ldb * l_n * i_micro_kernel_config->datatype_size_in;
        l_b_offset += (l_pass == 0) ? 2 : 0; /* we are emulating AVX512 VDPBF16PS */

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      LIBXSMM_X86_INSTR_VBCSTNEBF162PS,
                                      i_gp_reg_mapping->gp_reg_b,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_b_offset,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 1, 0 );
      }
      if ( l_pass == 1 ) {
        l_b_offset = i_micro_kernel_config->datatype_size_in * l_k_pack_factor;

        libxsmm_x86_instruction_alu_imm( io_generated_code,
                                      i_micro_kernel_config->alu_add_instruction,
                                      i_gp_reg_mapping->gp_reg_b,
                                      l_b_offset );
      }
    }

    /* load column vectors of A and multiply with all broadcasted row entries of B */
    for ( l_m = 0; l_m < l_m_blocking; l_m++ ) {
      if ( (i_micro_kernel_config->use_masking_a_c == 0) || ((i_micro_kernel_config->use_masking_a_c == 1) && (l_m < (l_m_blocking-1))) ) {
        unsigned int l_ld_instruction = (l_pass == 0) ? LIBXSMM_X86_INSTR_VCVTNEOBF162PS : LIBXSMM_X86_INSTR_VCVTNEEBF162PS;
        libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                     l_ld_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     LIBXSMM_X86_GP_REG_UNDEF, 0,
                                     (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                     i_micro_kernel_config->vector_name,
                                     i_n_blocking, 0, 0, 0 );

        /* post increment early */
        if ( (l_m == (l_m_blocking-1)) && (l_pass == 1) ) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
                                           i_micro_kernel_config->alu_add_instruction,
                                           i_gp_reg_mapping->gp_reg_a,
                                           (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
        }

        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          /* issue fma */
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    i_micro_kernel_config->vmul_instruction,
                                                    i_micro_kernel_config->vector_name,
                                                    i_n_blocking,
                                                    l_n,
                                                    l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      } else {
        for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code, i_micro_kernel_config->instruction_set, LIBXSMM_X86_INSTR_VMOVUPS,
                                            i_gp_reg_mapping->gp_reg_help_0, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, 'y', i_n_blocking, 0, 0, 0 );

          libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                                       i_micro_kernel_config->a_vmove_instruction,
                                       i_gp_reg_mapping->gp_reg_a,
                                       LIBXSMM_X86_GP_REG_UNDEF, 0,
                                       (i_micro_kernel_config->datatype_size_in) * (i_micro_kernel_config->vector_length) * l_m * l_k_pack_factor,
                                       i_micro_kernel_config->vector_name,
                                       i_n_blocking, ( l_m == (l_m_blocking - 1) ) ? i_micro_kernel_config->use_masking_a_c : 0, i_n_blocking, 0 );

          /* in the firs pass we conver the higher BF16 values to FP32; in the second the lower Bf16 values */
          if ( l_pass == 0 ) {
            libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPBLENDW,
                                                           i_micro_kernel_config->vector_name,
                                                           i_n_blocking, l_n, i_n_blocking, 0xaa );
          } else {
            libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I,
                                                           i_micro_kernel_config->vector_name,
                                                           i_n_blocking, i_n_blocking, 16 );
          }

          /* post increment early */
          if ( (l_m == (l_m_blocking-1)) && (l_n == (i_n_blocking-1)) && (l_pass == 1) ) {
            libxsmm_x86_instruction_alu_imm( io_generated_code,
                                             i_micro_kernel_config->alu_add_instruction,
                                             i_gp_reg_mapping->gp_reg_a,
                                             (long long)i_xgemm_desc->lda*i_micro_kernel_config->datatype_size_in*l_k_pack_factor );
          }

          /* issue fma */
          libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                                    i_micro_kernel_config->vmul_instruction,
                                                    i_micro_kernel_config->vector_name,
                                                    i_n_blocking,
                                                    l_n,
                                                    l_vec_reg_acc_start + l_m + (l_m_blocking * l_n) );
        }
      }
    }
  }
}

