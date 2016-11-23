/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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

#include "generator_spgemm_csr_asparse_reg.h"
#include "generator_x86_instructions.h"
#include "generator_gemm_common.h"
#include <libxsmm_macros.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_mmfunction_signature_asparse_reg( libxsmm_generated_code*         io_generated_code,
                                  const char*                     i_routine_name,
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
    if (0 != (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags)) {
      if (LIBXSMM_PREFETCH_NONE == i_xgemm_desc->prefetch) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const float* Bin, float* Cin) {\n", i_routine_name);
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const float* Bin, float* Cin, const float* Bin_prefetch, const float* Cin_prefetch) {\n", i_routine_name);
      }
    } else {
      if (LIBXSMM_PREFETCH_NONE == i_xgemm_desc->prefetch) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const double* Bin, double* Cin) {\n", i_routine_name);
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const double* Bin, double* Cin, const double* Bin_prefetch, const double* Cin_prefetch) {\n", i_routine_name);
      }
    }
  }

  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
}

LIBXSMM_INTERNAL_API_DEFINITION
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
#if 0
  unsigned int l_flop_count = 0;
#endif
  unsigned int l_unique;
  unsigned int l_hit;
  unsigned int l_n_i_blocking = 1;
#if 0
  unsigned int l_n_o_blocking = 1;
#endif
  unsigned int l_i_chunks = 1;
  double* l_unique_values = (double*)malloc(sizeof(double)*i_row_idx[i_xgemm_desc->m]);
  unsigned int* l_unique_pos = (unsigned int*)malloc(sizeof(unsigned int)*i_row_idx[i_xgemm_desc->m]);

  char l_new_code[512];
  int l_max_code_length = 511;
  int l_code_length = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* Let's figure out how many unique values we have */
  l_unique = 1;
  l_unique_values[0] = i_values[0];
  l_unique_pos[0] = 0;
  for ( l_m = 1; l_m < i_row_idx[i_xgemm_desc->m]; l_m++ ) {
    l_hit = 0;
    /* search for the value */
    for ( l_z = 0; l_z < l_unique; l_z++) {
      if ( /*l_unique_values[l_z] == i_values[l_m]*/!(l_unique_values[l_z] < i_values[l_m]) && !(l_unique_values[l_z] > i_values[l_m]) ) {
        l_unique_pos[l_m] = l_z;
        l_hit = 1;
      }
    }
    /* values was not found */
    if ( l_hit == 0 ) {
      l_unique_values[l_unique] = i_values[l_m];
      l_unique_pos[l_m] = l_unique;
      l_unique++;
    }
  }

  /* check that we have enough registers (N=20) for now */
  if ( l_unique > 31 ) {
    fprintf( stderr, "for reg version we right now can only have max. 31 unique non-zeros right now!" );
    exit(-1);
  }

  /* create a tempdata structure which contains the unique NNZ */
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  double A[%u];\n", l_unique);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  unsigned int l_n = 0;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  double* B;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  double* C;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  for ( l_z = 0; l_z < l_unique; l_z++) {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  A[%u] = %.20e;\n", l_z, l_unique_values[l_z]);
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* matching calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_X86_GP_REG_R9;
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

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, i_xgemm_desc, i_arch, 0 );

  /* inner chunck size */
  l_i_chunks = l_micro_kernel_config.vector_length*l_n_i_blocking;
  if ( i_xgemm_desc->n % l_i_chunks != 0 ) {
    fprintf(stderr, "n needs to be divisible by 96!\n");
    exit(-1);
  }

  /* generate the actuel kernel */
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  #pragma omp parallel for private(l_n,B,C)\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  for ( l_n = 0; l_n < %u; l_n+= %u ) {\n", (unsigned int)i_xgemm_desc->n, l_i_chunks);
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    B = ((double*)Bin)+l_n;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "    C = ((double*)Cin)+l_n;\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );

  /* load C into registers */
  for ( l_z = 0; l_z < l_unique; l_z++) {
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      l_micro_kernel_config.instruction_set,
                                      LIBXSMM_X86_INSTR_VBROADCASTSD,
                                      l_gp_reg_mapping.gp_reg_a,
                                      LIBXSMM_X86_GP_REG_UNDEF,
                                      0,
                                      l_micro_kernel_config.datatype_size*l_z,
                                      l_micro_kernel_config.vector_name,
                                      l_z,
                                      0,
                                      0 );
  }

  /* n loop */
#if 0
  libxsmm_x86_instruction_register_jump_label( io_generated_code, &l_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n_o_blocking );
#endif

  for ( l_m = 0; l_m < (unsigned int)i_xgemm_desc->m; l_m++ ) {
    l_row_elements = i_row_idx[l_m+1] - i_row_idx[l_m];
    if (l_row_elements > 0) {
      for ( l_n = 0; l_n < l_n_i_blocking; l_n++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          l_micro_kernel_config.instruction_set,
                                          l_micro_kernel_config.c_vmove_instruction,
                                          l_gp_reg_mapping.gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_m*i_xgemm_desc->ldc*l_micro_kernel_config.datatype_size +
                                            l_n*l_micro_kernel_config.datatype_size*l_micro_kernel_config.vector_length,
                                          l_micro_kernel_config.vector_name,
                                          l_unique+l_n, 0, 0 );

        libxsmm_x86_instruction_prefetch( io_generated_code,
                                          LIBXSMM_X86_INSTR_PREFETCHT2,
                                          l_gp_reg_mapping.gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_m*i_xgemm_desc->ldc*l_micro_kernel_config.datatype_size +
                                            (l_n+1)*l_micro_kernel_config.datatype_size*l_micro_kernel_config.vector_length );
      }
    }
    for ( l_z = 0; l_z < l_row_elements; l_z++ ) {
      /* check k such that we just use columns which actually need to be multiplied */
      for ( l_n = 0; l_n < l_n_i_blocking; l_n++ ) {
        libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                 l_micro_kernel_config.instruction_set,
                                                 l_micro_kernel_config.vmul_instruction,
                                                 0,
                                                 l_gp_reg_mapping.gp_reg_b,
                                                 LIBXSMM_X86_GP_REG_UNDEF,
                                                 0,
                                                 i_column_idx[i_row_idx[l_m] + l_z]*i_xgemm_desc->ldb*l_micro_kernel_config.datatype_size +
                                                   l_n*l_micro_kernel_config.datatype_size*l_micro_kernel_config.vector_length,
                                                 l_micro_kernel_config.vector_name,
                                                 l_unique_pos[i_row_idx[l_m] + l_z],
                                                 l_unique+l_n );

          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            LIBXSMM_X86_INSTR_PREFETCHT2,
                                            l_gp_reg_mapping.gp_reg_b,
                                            LIBXSMM_X86_GP_REG_UNDEF,
                                            0,
                                            i_column_idx[i_row_idx[l_m] + l_z]*i_xgemm_desc->ldb*l_micro_kernel_config.datatype_size +
                                              (l_n+1)*l_micro_kernel_config.datatype_size*l_micro_kernel_config.vector_length );
      }
    }
    if (l_row_elements > 0) {
      for ( l_n = 0; l_n < l_n_i_blocking; l_n++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          l_micro_kernel_config.instruction_set,
                                          l_micro_kernel_config.c_vmove_instruction,
                                          l_gp_reg_mapping.gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_m*i_xgemm_desc->ldc*l_micro_kernel_config.datatype_size +
                                            l_n*l_micro_kernel_config.datatype_size*l_micro_kernel_config.vector_length,
                                          l_micro_kernel_config.vector_name,
                                          l_unique+l_n, 0, 1 );
      }
    }
  }

  /* close n loop */
#if 0
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n_o_blocking );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );
#endif

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );

  /* close loop in C */
  l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "  }\n");
  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

  free(l_unique_values);
  free(l_unique_pos);
}

