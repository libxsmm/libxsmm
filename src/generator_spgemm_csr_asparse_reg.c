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
#include "generator_gemm_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_mmfunction_signature_asparse_reg( libxsmm_generated_code*        io_generated_code,
                                               const char*                    i_routine_name,
                                               const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  char l_new_code[512];
  int l_max_code_length = 511;
  int l_code_length = 0;

  if ( io_generated_code->code_type > 1 ) {
    return;
  } else if ( io_generated_code->code_type == 1 ) {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, ".global %s\n.type %s, @function\n%s:\n", i_routine_name, i_routine_name, i_routine_name);
  } else {
    /* selecting the correct signature */
    if (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      if (LIBXSMM_GEMM_PREFETCH_NONE == i_xgemm_desc->prefetch) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const float* A, const float* B, float* C) {\n", i_routine_name);
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const float* A, const float* B, float* C, const float* A_prefetch, const float* B_prefetch, const float* C_prefetch) {\n", i_routine_name);
      }
    } else {
      if (LIBXSMM_GEMM_PREFETCH_NONE == i_xgemm_desc->prefetch) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const double* A, const double* B, double* C) {\n", i_routine_name);
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const double* A, const double* B, double* C, const double* A_prefetch, const double* B_prefetch, const double* C_prefetch) {\n", i_routine_name);
      }
    }
  }

  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
}

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csr_asparse_reg( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                               const char*                     i_arch,
                                               const unsigned int*             i_row_idx,
                                               const unsigned int*             i_column_idx,
                                               const double*                   i_values ) {
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_z;
  unsigned int l_row_elements;
  unsigned int l_unique;
  unsigned int l_reg_num;
  unsigned int l_hit;
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
  if ( strcmp(i_arch, "knl") == 0 ) {
    io_generated_code->arch = LIBXSMM_X86_AVX512_MIC;
  } else if ( strcmp(i_arch, "knm") == 0 ) {
    io_generated_code->arch = LIBXSMM_X86_AVX512_KNM;
  } else if ( strcmp(i_arch, "skx") == 0 ) {
    io_generated_code->arch = LIBXSMM_X86_AVX512_CORE;
  } else if ( strcmp(i_arch, "clx") == 0 ) {
    io_generated_code->arch = LIBXSMM_X86_AVX512_CLX;
  } else if ( strcmp(i_arch, "cpx") == 0 ) {
    io_generated_code->arch = LIBXSMM_X86_AVX512_CPX;
  } else if ( strcmp(i_arch, "spr") == 0 ) {
    io_generated_code->arch = LIBXSMM_X86_AVX512_SPR;
  } else if ( strcmp(i_arch, "hsw") == 0 ) {
    io_generated_code->arch = LIBXSMM_X86_AVX2;
  } else {
    free(l_unique_values); free(l_unique_pos); free(l_unique_sgn);
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
    l_prefetch = 1;

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
  l_unique = 1;
  l_unique_values[0] = fabs(i_values[0]);
  l_unique_pos[0] = 0;
  l_unique_sgn[0] = (i_values[0] > 0) ? 1 : -1;
  for ( l_m = 1; l_m < l_n_row_idx; l_m++ ) {
    l_hit = 0;
    /* search for the value */
    for ( l_z = 0; l_z < l_unique; l_z++) {
      if ( /*l_unique_values[l_z] == i_values[l_m]*/!(l_unique_values[l_z] < fabs(i_values[l_m])) && !(l_unique_values[l_z] > fabs(i_values[l_m])) ) {
        l_unique_pos[l_m] = l_z;
        l_hit = 1;
      }
    }
    /* value was not found */
    if ( l_hit == 0 ) {
      l_unique_values[l_unique] = fabs(i_values[l_m]);
      l_unique_pos[l_m] = l_unique;
      l_unique++;
    }
    l_unique_sgn[l_m] = (i_values[l_m] > 0) ? 1 : -1;
  }

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

        /* only prefetch if we do temporal stores */
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
