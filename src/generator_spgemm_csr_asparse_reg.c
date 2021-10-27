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
#include "generator_spgemm_csr_asparse_reg.h"
#include "generator_x86_instructions.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common.h"
#include "generator_gemm_common_aarch64.h"
#include "libxsmm_main.h"


LIBXSMM_API_INTERN
void libxsmm_analyse_sparse_nnz( unsigned int   i_n_row_idx,
                                 const double*  i_values,
                                 unsigned int*  o_unique,
                                 double*        o_unique_values,
                                 unsigned int*  o_unique_pos,
                                 int*           o_unique_sgn ) {
  unsigned int l_unique = 1;
  unsigned int l_hit, l_m, l_z;

  o_unique_values[0] = fabs(i_values[0]);
  o_unique_pos[0] = 0;
  o_unique_sgn[0] = (i_values[0] > 0) ? 1 : -1;
  for ( l_m = 1; l_m < i_n_row_idx; l_m++ ) {
    l_hit = 0;
    /* search for the value */
    for ( l_z = 0; l_z < l_unique; l_z++ ) {
      if ( !(o_unique_values[l_z] < fabs(i_values[l_m])) && !(o_unique_values[l_z] > fabs(i_values[l_m])) ) {
        o_unique_pos[l_m] = l_z;
        l_hit = 1;
      }
    }
    /* value was not found */
    if ( !l_hit ) {
      o_unique_values[l_unique] = fabs(i_values[l_m]);
      o_unique_pos[l_m] = l_unique++;
    }
    o_unique_sgn[l_m] = (i_values[l_m] > 0) ? 1 : -1;
  }
  *o_unique = l_unique;
}

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csr_asparse_reg_x86( libxsmm_generated_code*         io_generated_code,
                                                   const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                   const unsigned int*             i_row_idx,
                                                   const unsigned int*             i_column_idx,
                                                   const double*                   i_values ) {
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_z;
  unsigned int l_row_elements;
  unsigned int l_unique;
  unsigned int l_reg_num;
  unsigned int l_n_blocking;
  unsigned int l_n_row_idx = i_row_idx[i_xgemm_desc->m];
  double *const l_unique_values = (double*)(0 != l_n_row_idx ? malloc(sizeof(double) * l_n_row_idx) : NULL);
  unsigned int *const l_unique_pos = (unsigned int*)(0 != l_n_row_idx ? malloc(sizeof(unsigned int) * l_n_row_idx) : NULL);
  int *const l_unique_sgn = (int*)(0 != l_n_row_idx ? malloc(sizeof(int) * l_n_row_idx) : NULL);
  double l_code_const_dp[8];
  float l_code_const_fp[16];
  unsigned int l_const_perm_ops[16];

  unsigned int l_fp64 = LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype );
  unsigned int l_breg_unique, l_preg_unique, l_psreg_unique;
  unsigned int l_base_acc_reg, l_base_perm_reg, l_bcast_reg;
  int l_prefetch;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* check if mallocs were successful */
  if ( 0 == l_unique_values || 0 == l_unique_pos || 0 == l_unique_sgn ) {
    free(l_unique_values); free(l_unique_pos); free(l_unique_sgn);
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_ALLOC_DATA );
    return;
  }

  /* Check that the arch is supported */
  if ( io_generated_code->arch < LIBXSMM_X86_AVX2 ) {
    free( l_unique_values ); free( l_unique_pos ); free( l_unique_sgn );
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* Define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc, 0 );

  /* Inner chunk size */
  if ( i_xgemm_desc->n == l_micro_kernel_config.vector_length ) {
    l_n_blocking = 1;
  } else if ( i_xgemm_desc->n == 2*l_micro_kernel_config.vector_length ) {
    l_n_blocking = 2;
  } else {
      free(l_unique_values); free(l_unique_pos); free(l_unique_sgn);
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
      return;
  }

  /* Init config */
  if ( io_generated_code->arch == LIBXSMM_X86_AVX2 ) {
    l_breg_unique = 16 - l_n_blocking;
    l_base_acc_reg = 16 - l_n_blocking;
    l_prefetch = 0;

    l_preg_unique = l_psreg_unique = 0;
    l_base_perm_reg = l_bcast_reg = (unsigned int)-1;
  } else {
    l_breg_unique = 32 - l_n_blocking;
    l_base_acc_reg = 32 - l_n_blocking;
    l_bcast_reg = l_base_acc_reg - 1;
    if ( (io_generated_code->arch == LIBXSMM_X86_AVX512_MIC) ||
         (io_generated_code->arch == LIBXSMM_X86_AVX512_KNM)    ) {
      l_prefetch = 1;
    } else {
      l_prefetch = 0;
    }

    if ( l_fp64 ) {
      l_preg_unique = (32 - l_n_blocking - 1 - 8)*8;
      l_psreg_unique = (32 - l_n_blocking - 1)*8;
      l_base_perm_reg = l_bcast_reg - 8;
    } else {
      l_preg_unique = (32 - l_n_blocking - 1 - 16)*16;
      l_psreg_unique = (32 - l_n_blocking - 1)*16;
      l_base_perm_reg = l_bcast_reg - 16;
    }
  }

  /* prerequisite */
  assert(0 != i_values);

  /* Let's figure out how many unique values we have */
  libxsmm_analyse_sparse_nnz( l_n_row_idx, i_values, &l_unique,
                              l_unique_values, l_unique_pos, l_unique_sgn );

  /* check that there are not too many unique values */
  if ( l_unique > l_breg_unique && l_unique > l_preg_unique && l_unique > l_psreg_unique ) {
    free(l_unique_values); free(l_unique_pos); free(l_unique_sgn);
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNIQUE_VAL );
    return;
  }

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
  /* TODO: full support for Windows calling convention */
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
/*  l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_X86_GP_REG_UNDEF;*/
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
/*  l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_X86_GP_REG_R9;*/
#endif
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );

  /*
   * load A into registers
   * pre-broadcast if possible, otherwise load for run-time broadcasting
   */
  if (l_unique <= l_breg_unique ) {
    /* pre-broadcast A values into registers */
    for ( l_z = 0; l_z < l_unique; l_z++) {
      char l_id[65];
      LIBXSMM_SNPRINTF(l_id, 64, "%u", l_z);
      if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
        for ( l_m = 0; l_m < 8; l_m++) {
          l_code_const_dp[l_m] = l_unique_values[l_z];
        }
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                            (unsigned char*)l_code_const_dp,
                                                            l_id,
                                                            l_micro_kernel_config.vector_name,
                                                            l_z );
      } else {
        for ( l_m = 0; l_m < 16; l_m++) {
          l_code_const_fp[l_m] = (float)l_unique_values[l_z];
        }
        libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
                                                            (unsigned char*)l_code_const_fp,
                                                            l_id,
                                                            l_micro_kernel_config.vector_name,
                                                            l_z );
      }
    }
  } else {
    /* load packed A into registers */
    l_z = 0;
    l_reg_num = 0;
    while (l_z < l_unique) {
      char l_id[65];
      LIBXSMM_SNPRINTF(l_id, 64, "%u", l_reg_num);
      l_m = 0;

      if ( l_fp64 ) {
        while (l_z < l_unique && l_m < 8) {
          l_code_const_dp[l_m++] = l_unique_values[l_z++];
        }
        libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code,
                                                            (unsigned char*)l_code_const_dp,
                                                            l_id,
                                                            l_micro_kernel_config.vector_name,
                                                            l_reg_num++ );
      } else {
        while (l_z < l_unique && l_m < 16) {
          l_code_const_fp[l_m++] = (float)l_unique_values[l_z++];
        }
        libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code,
                                                            (unsigned char*)l_code_const_fp,
                                                            l_id,
                                                            l_micro_kernel_config.vector_name,
                                                            l_reg_num++ );
      }
    }

    /* load permute operands into registers if space is available (otherwise they are read from memory) */
    if ( l_fp64 && l_unique <= l_preg_unique ) {
      for (l_reg_num = l_base_perm_reg; l_reg_num < l_base_perm_reg + 8; l_reg_num++ ) {
        char l_id[65];
        LIBXSMM_SNPRINTF(l_id, 64, "%u", l_reg_num);
        l_m = 0;
        /* repeat pattern to select 64-bits using vpermd */
        while (l_m < 16) {
            l_const_perm_ops[l_m++] = (l_reg_num - l_base_perm_reg)*2;
            l_const_perm_ops[l_m++] = (l_reg_num - l_base_perm_reg)*2 + 1;
        }
        libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code,
                                                            (unsigned char*)l_const_perm_ops,
                                                            l_id,
                                                            l_micro_kernel_config.vector_name,
                                                            l_reg_num );
      }
    } else if ( !l_fp64 && l_unique <= l_preg_unique ) {
      for (l_reg_num = l_base_perm_reg; l_reg_num<l_base_perm_reg + 16; l_reg_num++ ) {
        char l_id[65];
        LIBXSMM_SNPRINTF(l_id, 64, "%u", l_reg_num);
        l_m = 0;
        /* repeat pattern to select 32-bits using vpermd */
        while (l_m < 16) {
            l_const_perm_ops[l_m++] = (l_reg_num - l_base_perm_reg);
        }
        libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code,
                                                            (unsigned char*)l_const_perm_ops,
                                                            l_id,
                                                            l_micro_kernel_config.vector_name,
                                                            l_reg_num );
      }
    }
  }

  /* n loop */
#if 0
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n_blocking );
#endif

  for ( l_m = 0; l_m < (unsigned int)i_xgemm_desc->m; l_m++ ) {
    l_row_elements = i_row_idx[l_m+1] - i_row_idx[l_m];
    if (l_row_elements > 0) {
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        /* load C or reset to 0 depending on beta */
        if ( 0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags) ) { /* Beta=1 */
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            l_micro_kernel_config.instruction_set,
                                            l_micro_kernel_config.c_vmove_instruction,
                                            l_gp_reg_mapping.gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_m*i_xgemm_desc->ldc*l_micro_kernel_config.datatype_size_out +
                                              l_n*l_micro_kernel_config.datatype_size_out*l_micro_kernel_config.vector_length,
                                            l_micro_kernel_config.vector_name,
                                            l_base_acc_reg + l_n, 0, 1, 0 );
        } else {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   l_micro_kernel_config.instruction_set,
                                                   l_micro_kernel_config.vxor_instruction,
                                                   l_micro_kernel_config.vector_name,
                                                   l_base_acc_reg + l_n,
                                                   l_base_acc_reg + l_n,
                                                   l_base_acc_reg + l_n );
        }

        /* only prefetch if we're not doing temporal stores */
        if ( l_prefetch && (LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT & i_xgemm_desc->flags) == 0 ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT2,
                                            l_gp_reg_mapping.gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_m*i_xgemm_desc->ldc*l_micro_kernel_config.datatype_size_out +
                                              (l_n+1)*l_micro_kernel_config.datatype_size_out*l_micro_kernel_config.vector_length );
        }
      }
    }
    for ( l_z = 0; l_z < l_row_elements; l_z++ ) {
      /* check k such that we just use columns which actually need to be multiplied */
      const unsigned int u = i_row_idx[l_m] + l_z;
      unsigned int l_unique_reg, fma_instruction;
      LIBXSMM_ASSERT(u < l_n_row_idx);

      /* select the correct FMA instruction */
      if ( l_fp64 ) {
        fma_instruction = (l_unique_sgn[u] == 1) ? LIBXSMM_X86_INSTR_VFMADD231PD : LIBXSMM_X86_INSTR_VFNMADD231PD;
      } else {
        fma_instruction = (l_unique_sgn[u] == 1) ? LIBXSMM_X86_INSTR_VFMADD231PS : LIBXSMM_X86_INSTR_VFNMADD231PS;
      }

      /* broadcast unique element of A if not in pre-broadcast mode */
      if (l_unique > l_breg_unique ) {
        if ( l_fp64 ) {
          /* load permute selector operand if not stored in registers */
          if ( l_unique > l_preg_unique ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              l_micro_kernel_config.instruction_set,
                                              l_micro_kernel_config.a_vmove_instruction,
                                              l_gp_reg_mapping.gp_reg_a,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              (l_unique_pos[u] % 8)*64,
                                              l_micro_kernel_config.vector_name,
                                              l_bcast_reg, 0, 1, 0 );

            libxsmm_x86_instruction_vec_compute_reg(io_generated_code,
                                                    l_micro_kernel_config.instruction_set,
                                                    LIBXSMM_X86_INSTR_VPERMD,
                                                    l_micro_kernel_config.vector_name,
                                                    l_unique_pos[u] / 8,
                                                    l_bcast_reg,
                                                    l_bcast_reg);

          /* permute selector operand already in register */
          } else {
            libxsmm_x86_instruction_vec_compute_reg(io_generated_code,
                                                    l_micro_kernel_config.instruction_set,
                                                    LIBXSMM_X86_INSTR_VPERMD,
                                                    l_micro_kernel_config.vector_name,
                                                    l_unique_pos[u] / 8,
                                                    l_base_perm_reg + l_unique_pos[u] % 8,
                                                    l_bcast_reg);
          }
        } else {
          /* load permute selector operand if not stored in registers */
          if ( l_unique > l_preg_unique ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                                              l_micro_kernel_config.instruction_set,
                                              l_micro_kernel_config.a_vmove_instruction,
                                              l_gp_reg_mapping.gp_reg_a,
                                              LIBXSMM_X86_GP_REG_UNDEF, 0,
                                              (l_unique_pos[u] % 16)*64,
                                              l_micro_kernel_config.vector_name,
                                              l_bcast_reg, 0, 1, 0 );

            libxsmm_x86_instruction_vec_compute_reg(io_generated_code,
                                                    l_micro_kernel_config.instruction_set,
                                                    LIBXSMM_X86_INSTR_VPERMD,
                                                    l_micro_kernel_config.vector_name,
                                                    l_unique_pos[u] / 16,
                                                    l_bcast_reg,
                                                    l_bcast_reg);

          /* permute selector operand already in register */
          } else {
            libxsmm_x86_instruction_vec_compute_reg(io_generated_code,
                                                    l_micro_kernel_config.instruction_set,
                                                    LIBXSMM_X86_INSTR_VPERMD,
                                                    l_micro_kernel_config.vector_name,
                                                    l_unique_pos[u] / 16,
                                                    l_base_perm_reg + l_unique_pos[u] % 16,
                                                    l_bcast_reg);
          }
        }
      }

      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        /* select correct register depending on mode */
        l_unique_reg = l_unique > 31 ? l_bcast_reg : l_unique_pos[u];

        libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                 l_micro_kernel_config.instruction_set,
                                                 fma_instruction,
                                                 0,
                                                 l_gp_reg_mapping.gp_reg_b,
                                                 LIBXSMM_X86_GP_REG_UNDEF,
                                                 0,
                                                 i_column_idx[u]*i_xgemm_desc->ldb*l_micro_kernel_config.datatype_size_in +
                                                   l_n*l_micro_kernel_config.datatype_size_in*l_micro_kernel_config.vector_length,
                                                 l_micro_kernel_config.vector_name,
                                                 l_unique_reg,
                                                 l_base_acc_reg + l_n );

          if ( l_prefetch )
            libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT2,
                                            l_gp_reg_mapping.gp_reg_b,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            i_column_idx[u]*i_xgemm_desc->ldb*l_micro_kernel_config.datatype_size_in +
                                              (l_n+1)*l_micro_kernel_config.datatype_size_in*l_micro_kernel_config.vector_length );
      }
    }
    if (l_row_elements > 0) {
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        unsigned int l_store_instruction = 0;
        if ((LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT & i_xgemm_desc->flags) > 0) {
          if ( l_fp64 ) {
            l_store_instruction = LIBXSMM_X86_INSTR_VMOVNTPD;
          } else {
            l_store_instruction = LIBXSMM_X86_INSTR_VMOVNTPS;
          }
        } else {
          l_store_instruction = l_micro_kernel_config.c_vmove_instruction;
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          l_micro_kernel_config.instruction_set,
                                          l_store_instruction,
                                          l_gp_reg_mapping.gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_m*i_xgemm_desc->ldc*l_micro_kernel_config.datatype_size_out +
                                            l_n*l_micro_kernel_config.datatype_size_out*l_micro_kernel_config.vector_length,
                                          l_micro_kernel_config.vector_name,
                                          l_base_acc_reg + l_n, 0, 0, 1 );
      }
    }
  }

  /* close n loop */
#if 0
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n_blocking );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );
#endif

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );

  free(l_unique_values);
  free(l_unique_pos);
  free(l_unique_sgn);
}

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csr_asparse_reg_aarch64_neon( libxsmm_generated_code*         io_generated_code,
                                                            const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                            const unsigned int*             i_row_idx,
                                                            const unsigned int*             i_column_idx,
                                                            const double*                   i_values ) {
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_z;
  unsigned int l_row_elements;
  unsigned int l_unique;
  unsigned int l_n_blocking;
  unsigned int l_n_row_idx = i_row_idx[i_xgemm_desc->m];
  double *const l_unique_values = (double*)(0 != l_n_row_idx ? malloc(sizeof(double) * l_n_row_idx) : NULL);
  unsigned int *const l_unique_pos = (unsigned int*)(0 != l_n_row_idx ? malloc(sizeof(unsigned int) * l_n_row_idx) : NULL);
  int *const l_unique_sgn = (int*)(0 != l_n_row_idx ? malloc(sizeof(int) * l_n_row_idx) : NULL);
  const unsigned int l_fp64 = LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype );
  unsigned int l_reg_unique, l_base_acc_reg, l_base_ld_reg;

  const libxsmm_aarch64_asimd_tupletype l_tuplet = (l_fp64) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S;
  const libxsmm_aarch64_asimd_width l_width = (l_fp64) ? LIBXSMM_AARCH64_ASIMD_WIDTH_D : LIBXSMM_AARCH64_ASIMD_WIDTH_S;
  const unsigned int l_values_per_reg = (l_fp64) ? 2 : 4;

  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* check if mallocs were successful */
  if ( 0 == l_unique_values || 0 == l_unique_pos || 0 == l_unique_sgn ) {
    free( l_unique_values ); free( l_unique_pos ); free( l_unique_sgn );
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_ALLOC_DATA );
    return;
  }

  /* Inner chunk size */
  if ( i_xgemm_desc->n == l_values_per_reg ) {
    l_n_blocking = 1;
  } else if ( i_xgemm_desc->n == 2*l_values_per_reg ) {
    l_n_blocking = 2;
  } else if ( i_xgemm_desc->n == 4*l_values_per_reg ) {
    l_n_blocking = 4;
  } else {
    free( l_unique_values ); free( l_unique_pos ); free( l_unique_sgn );
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* Init config */
  l_reg_unique = l_values_per_reg*(32 - l_n_blocking - ((l_n_blocking > 1) ? 2 : 1));
  l_base_acc_reg = 32 - l_n_blocking;
  l_base_ld_reg = l_base_acc_reg - ((l_n_blocking > 1) ? 2 : 1);

  /* prerequisite */
  LIBXSMM_ASSERT(0 != i_values);

  /* Let's figure out how many unique values we have */
  libxsmm_analyse_sparse_nnz( l_n_row_idx, i_values, &l_unique,
                              l_unique_values, l_unique_pos, l_unique_sgn );

  /* check that there are not too many unique values */
  if ( l_unique > l_reg_unique ) {
    free( l_unique_values ); free( l_unique_pos ); free( l_unique_sgn );
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNIQUE_VAL );
    return;
  }

  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_AARCH64_GP_REG_X4;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_AARCH64_GP_REG_X7;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_AARCH64_GP_REG_X8;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X9;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X11;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_X27;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_AARCH64_GP_REG_X28;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_X29;

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xe0f );

  /* load A into registers */
  for ( l_z = 0; l_z < l_unique; l_z++) {
    unsigned long long l_imm;

    if ( l_fp64 ) {
      union { double f; unsigned long long i; } u;
      u.f = l_unique_values[l_z];
      l_imm = u.i;
    } else {
      union { float f; unsigned int i; } u;
      u.f = (float) l_unique_values[l_z];
      l_imm = (unsigned long long) u.i;
    }

    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code,
                                               l_gp_reg_mapping.gp_reg_help_1,
                                               l_imm );
    libxsmm_aarch64_instruction_asimd_gpr_move( io_generated_code,
                                                LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V,
                                                l_gp_reg_mapping.gp_reg_help_1,
                                                l_z / l_values_per_reg,
                                                l_z % l_values_per_reg,
                                                l_width );
  }

  for ( l_m = 0; l_m < (unsigned int)i_xgemm_desc->m; l_m++ ) {
    l_row_elements = i_row_idx[l_m + 1] - i_row_idx[l_m];
    if ( 0 == l_row_elements ) {
      continue;
    }

    /* Compute the address of the relevant row in C */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_3,
                                                   l_gp_reg_mapping.gp_reg_help_5,
                                                   l_m*i_xgemm_desc->ldc*(16 / l_values_per_reg) );

    /* Beta = 0; zero the accumulators */
    if ( LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags ) {
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                   l_base_acc_reg + l_n, l_base_acc_reg + l_n, 0, l_base_acc_reg + l_n,
                                                   LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
      }
    }
    /* Beta = 1; load C into the accumulators */
    else {
      if ( 1 == l_n_blocking ) {
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF,
                                                l_gp_reg_mapping.gp_reg_help_5, 0, 0,
                                                l_base_acc_reg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
      } else {
        for ( l_n = 0; l_n < l_n_blocking; l_n += 2 ) {
          libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF,
                                                       l_gp_reg_mapping.gp_reg_help_5, 16*l_n,
                                                       l_base_acc_reg + l_n, l_base_acc_reg + l_n + 1,
                                                       LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        }
      }
    }

    for ( l_z = 0; l_z < l_row_elements; l_z++ ) {
      const unsigned int u = i_row_idx[l_m] + l_z;
      unsigned int l_unique_reg, fma_instruction;
      LIBXSMM_ASSERT(u < l_n_row_idx); /* mute issue pointed out by Clang's static analysis */
      l_unique_reg = l_unique_pos[u];
      fma_instruction = (l_unique_sgn[u] == 1) ? LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V : LIBXSMM_AARCH64_INSTR_ASIMD_FMLS_E_V;

      if ( 1 == l_n_blocking ) {
        libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_gp_reg_mapping.gp_reg_help_1,
                                                   i_column_idx[u]*i_xgemm_desc->ldb*(16 / l_values_per_reg) );
        libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R,
                                                l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_1, 0,
                                                l_base_ld_reg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, fma_instruction,
                                                   l_base_ld_reg,
                                                   l_unique_reg / l_values_per_reg,
                                                   l_unique_reg % l_values_per_reg,
                                                   l_base_acc_reg, l_tuplet );
      } else {
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_2,
                                                       l_gp_reg_mapping.gp_reg_help_3, i_column_idx[u]*i_xgemm_desc->ldb*(16 / l_values_per_reg) );
        for ( l_n = 0; l_n < l_n_blocking; l_n += 2 ) {
          libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF,
                                                       l_gp_reg_mapping.gp_reg_help_3, 16*l_n,
                                                       l_base_ld_reg, l_base_ld_reg + 1,
                                                       LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, fma_instruction,
                                                     l_base_ld_reg,
                                                     l_unique_reg / l_values_per_reg,
                                                     l_unique_reg % l_values_per_reg,
                                                     l_base_acc_reg + l_n, l_tuplet );
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, fma_instruction,
                                                     l_base_ld_reg + 1,
                                                     l_unique_reg / l_values_per_reg,
                                                     l_unique_reg % l_values_per_reg,
                                                     l_base_acc_reg + l_n + 1,
                                                     l_tuplet );
        }
      }
    }

    if ( 1 == l_n_blocking ) {
      libxsmm_aarch64_instruction_asimd_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF,
                                              l_gp_reg_mapping.gp_reg_help_5, 0, 0,
                                              l_base_acc_reg, LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
    } else {
      for ( l_n = 0; l_n < l_n_blocking; l_n += 2 ) {
        libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF,
                                                     l_gp_reg_mapping.gp_reg_help_5, 16*l_n,
                                                     l_base_acc_reg + l_n, l_base_acc_reg + l_n + 1,
                                                     LIBXSMM_AARCH64_ASIMD_WIDTH_Q );
      }
    }
  }

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );

  free( l_unique_values );
  free( l_unique_pos );
  free( l_unique_sgn );
}

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csr_asparse_reg_aarch64_sve( libxsmm_generated_code*         io_generated_code,
                                                           const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                           const unsigned int*             i_row_idx,
                                                           const unsigned int*             i_column_idx,
                                                           const double*                   i_values ) {
  unsigned int l_m, l_n, l_z;
  unsigned int l_row_elements;
  unsigned int l_unique;
  unsigned int l_n_blocking;
  unsigned int l_n_row_idx = i_row_idx[i_xgemm_desc->m];
  double *const l_unique_values = (double*)(0 != l_n_row_idx ? malloc(sizeof(double) * l_n_row_idx) : NULL);
  unsigned int *const l_unique_pos = (unsigned int*)(0 != l_n_row_idx ? malloc(sizeof(unsigned int) * l_n_row_idx) : NULL);
  int *const l_unique_sgn = (int*)(0 != l_n_row_idx ? malloc(sizeof(int) * l_n_row_idx) : NULL);
  unsigned int l_reg_unique, l_base_acc_reg, l_ld_reg;

  const unsigned int l_fp64 = LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype );
  const unsigned int l_fbytes = (l_fp64) ? 8 : 4;
  unsigned int l_npacked_reg, l_npacked_values_per_reg;

  libxsmm_aarch64_sve_type l_svet;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* check if mallocs were successful */
  if ( 0 == l_unique_values || 0 == l_unique_pos || 0 == l_unique_sgn ) {
    free( l_unique_values ); free( l_unique_pos ); free( l_unique_sgn );
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CSR_ALLOC_DATA );
    return;
  }

  /* Define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );

  /* Inner chunk size */
  if ( i_xgemm_desc->n == l_micro_kernel_config.vector_length ) {
    l_n_blocking = 1;
  } else if ( i_xgemm_desc->n == 2*l_micro_kernel_config.vector_length ) {
    l_n_blocking = 2;
  } else if ( i_xgemm_desc->n == 4*l_micro_kernel_config.vector_length ) {
    l_n_blocking = 4;
  } else {
    free( l_unique_values ); free( l_unique_pos ); free( l_unique_sgn );
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* Init config */
  if ( l_fp64 ) {
    l_npacked_reg = 0; /*16;*/
    l_npacked_values_per_reg = 2;
    l_svet = LIBXSMM_AARCH64_SVE_TYPE_D;
  } else {
    l_npacked_reg = 0; /*8;*/
    l_npacked_values_per_reg = 4;
    l_svet = LIBXSMM_AARCH64_SVE_TYPE_S;
  }

  l_reg_unique = l_npacked_reg*l_npacked_values_per_reg + (32 - l_n_blocking - 1 - l_npacked_reg);
  l_base_acc_reg = 32 - l_n_blocking;
  l_ld_reg = l_base_acc_reg - 1;

  /* prerequisite */
  LIBXSMM_ASSERT(0 != i_values);

  /* Let's figure out how many unique values we have */
  libxsmm_analyse_sparse_nnz( l_n_row_idx, i_values, &l_unique,
                              l_unique_values, l_unique_pos, l_unique_sgn );

  /* check that there are not too many unique values */
  if ( l_unique > l_reg_unique ) {
    free( l_unique_values ); free( l_unique_pos ); free( l_unique_sgn );
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNIQUE_VAL );
    return;
  }

  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_AARCH64_GP_REG_X4;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_AARCH64_GP_REG_X7;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_AARCH64_GP_REG_X8;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X9;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X11;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_X27;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_AARCH64_GP_REG_X28;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_X29;

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xe0f );

  /* Load A (l_n) into registers (l_z) */
  for ( l_n = l_z = 0; l_n < l_unique; l_z++) {
    double l_rvals_d[2] = {};
    float l_rvals_f[4] = {};
    void *imm_arr;

    /* Broadcast the current unique value to the register */
    if ( l_fp64 ) {
      l_rvals_d[0] = l_rvals_d[1] = l_unique_values[l_n++];
      imm_arr = (void*)l_rvals_d;
    } else {
      l_rvals_f[0] = l_rvals_f[1] = l_rvals_f[2] = l_rvals_f[3] = (float) l_unique_values[l_n++];
      imm_arr = (void*)l_rvals_f;
    }

    /* See if we can pack any more values in */
    if ( l_z < l_npacked_reg ) {
      if ( l_fp64 && l_n < l_unique ) {
        l_rvals_d[1] = l_unique_values[l_n++];
      } else {
        for ( l_m = 1; l_m < 4 && l_n < l_unique; l_m++ ) {
          l_rvals_f[l_m] = (float) l_unique_values[l_n++];
        }
      }
    }

    libxsmm_aarch64_instruction_sve_rep16bytes_const_to_vec( io_generated_code, l_z,
                                                             l_gp_reg_mapping.gp_reg_help_0, l_gp_reg_mapping.gp_reg_help_1,
                                                             LIBXSMM_AARCH64_SVE_REG_P0, imm_arr, 0 );
  }

  libxsmm_generator_set_p_register_aarch64_sve( io_generated_code, LIBXSMM_AARCH64_SVE_REG_P0,
                                                -1, l_gp_reg_mapping.gp_reg_help_0 );


  for ( l_m = 0; l_m < (unsigned int)i_xgemm_desc->m; l_m++ ) {
    l_row_elements = i_row_idx[l_m + 1] - i_row_idx[l_m];
    if ( 0 == l_row_elements ) {
      continue;
    }

    /* Compute the address of the relevant row in C */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   l_gp_reg_mapping.gp_reg_c, l_gp_reg_mapping.gp_reg_help_3,
                                                   l_gp_reg_mapping.gp_reg_help_5,
                                                   l_m*i_xgemm_desc->ldc*l_fbytes );

    /* Beta = 0; zero the accumulators */
    if ( LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags ) {
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                 l_base_acc_reg + l_n, l_base_acc_reg + l_n, 0, l_base_acc_reg + l_n,
                                                 LIBXSMM_AARCH64_GP_REG_UNDEF, l_svet );
      }
    }
    /* Beta = 1; load C into the accumulators */
    else {
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                              l_gp_reg_mapping.gp_reg_help_5, 0, l_n,
                                              l_base_acc_reg + l_n, LIBXSMM_AARCH64_SVE_REG_P0 );
      }
    }

    for ( l_z = 0; l_z < l_row_elements; l_z++ ) {
      const unsigned int u = i_row_idx[l_m] + l_z;
      unsigned int l_unique_reg, l_idx, fma_instruction;
      LIBXSMM_ASSERT(u < l_n_row_idx);

      /* Constant is packed in its register */
      if ( l_unique_pos[u] < l_npacked_reg*l_npacked_values_per_reg ) {
        l_unique_reg = l_unique_pos[u] / l_npacked_values_per_reg;
        l_idx = l_unique_pos[u] % l_npacked_values_per_reg;
        fma_instruction = (l_unique_sgn[u] == 1) ? LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_I : LIBXSMM_AARCH64_INSTR_SVE_FMLS_V_I;
      /* Constant is broadcasted in its register */
      } else {
        l_unique_reg = l_unique_pos[u] - l_npacked_reg*(l_npacked_values_per_reg - 1);
        l_idx = 0;
        fma_instruction = (l_unique_sgn[u] == 1) ? LIBXSMM_AARCH64_INSTR_SVE_FMLA_V : LIBXSMM_AARCH64_INSTR_SVE_FMLS_V;
      }

      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                      l_gp_reg_mapping.gp_reg_b, l_gp_reg_mapping.gp_reg_help_2,
                                                      l_gp_reg_mapping.gp_reg_help_3, i_column_idx[u]*i_xgemm_desc->ldb*l_fbytes );
      for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
        libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF,
                                              l_gp_reg_mapping.gp_reg_help_3, 0, l_n,
                                              l_ld_reg, LIBXSMM_AARCH64_SVE_REG_P0 );
        libxsmm_aarch64_instruction_sve_compute ( io_generated_code, fma_instruction,
                                                  l_ld_reg, l_unique_reg, l_idx, l_base_acc_reg + l_n,
                                                  LIBXSMM_AARCH64_SVE_REG_P0, l_svet );
      }
    }

    for ( l_n = 0; l_n < l_n_blocking; l_n++ ) {
      libxsmm_aarch64_instruction_sve_move( io_generated_code, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF,
                                            l_gp_reg_mapping.gp_reg_help_5, 0, l_n,
                                            l_base_acc_reg + l_n, LIBXSMM_AARCH64_SVE_REG_P0 );
    }
  }

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );

  free( l_unique_values );
  free( l_unique_pos );
  free( l_unique_sgn );
}
