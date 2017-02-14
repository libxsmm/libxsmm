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

#include "generator_spgemm_csr_asparse_soa.h"
#include "generator_gemm_common.h"
#include "generator_x86_instructions.h"
#include <libxsmm_macros.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_spgemm_csr_asparse_soa( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                               const char*                     i_arch,
                                               const unsigned int*             i_row_idx,
                                               const unsigned int*             i_column_idx,
                                               const double*                   i_values ) {
  if ( strcmp(i_arch, "knl") == 0 ||
       strcmp(i_arch, "skx") == 0 ) {
    libxsmm_generator_spgemm_csr_asparse_soa_avx512( io_generated_code,
                                                     i_xgemm_desc,
                                                     i_arch,
                                                     i_row_idx,
                                                     i_column_idx,
                                                     i_values );
  } else {
    fprintf( stderr, "CSR + SOA is only available for AVX512 at this point" );
    exit(-1);
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_spgemm_csr_asparse_soa_avx512( libxsmm_generated_code*         io_generated_code,
                                                      const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                      const char*                     i_arch,
                                                      const unsigned int*             i_row_idx,
                                                      const unsigned int*             i_column_idx,
                                                      const double*                   i_values ) {
  unsigned int l_soa_width = 0;
  unsigned int l_gen_m_trips = 0;
  unsigned int l_a_is_dense = 0;
  unsigned int l_n_chunks = 0;
  unsigned int l_n_chunksize = 0;
  unsigned int l_n_remain = 0;
  unsigned int l_n_max_block = 28;

  libxsmm_micro_kernel_config l_micro_kernel_config = { 0 };
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* matching calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
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

  /* select soa width */
  if ( (LIBXSMM_GEMM_FLAG_F32PREC & i_xgemm_desc->flags) == 0 ) {
    l_soa_width = 8;
    l_micro_kernel_config.a_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSD;
  } else {
    l_soa_width = 16;
    l_micro_kernel_config.a_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
  }

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );

  /* test if we should generate a dense version */
  if ( i_row_idx[i_xgemm_desc->m] == (unsigned int)(i_xgemm_desc->m*i_xgemm_desc->k) ) {
    l_gen_m_trips = 1;
    l_a_is_dense = 1;
  } else {
    l_gen_m_trips = i_xgemm_desc->m;
    l_a_is_dense = 0;
  }

  /* calculate the chunk size of current columns to work on */
  l_n_chunks = ( (i_xgemm_desc->n % l_n_max_block) == 0 ) ? (i_xgemm_desc->n / l_n_max_block) : (i_xgemm_desc->n / l_n_max_block) + 1;
  l_n_chunksize = ( (i_xgemm_desc->n % l_n_chunks) == 0 ) ? (i_xgemm_desc->n / l_n_chunks) : (i_xgemm_desc->n / l_n_chunks) + 1;
  l_n_remain = ( ((i_xgemm_desc->n % l_n_chunks) == 0) || ((unsigned int)i_xgemm_desc->n <= l_n_max_block) ) ? 0 : 1;

  /* loop over blocks of n */
  libxsmm_x86_instruction_register_jump_label( io_generated_code, &l_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n_chunksize );

  /* do matix multiplicatoin for a block of N columns */
  libxsmm_generator_spgemm_csr_asparse_soa_m_loop_avx512( io_generated_code, i_xgemm_desc, &l_loop_label_tracker, &l_micro_kernel_config, &l_gp_reg_mapping, i_arch,
                                                            i_row_idx, i_column_idx, i_values,
                                                            l_soa_width, l_gen_m_trips, l_a_is_dense, l_n_chunksize );

  /* adjust B pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_b,
                                     l_micro_kernel_config.datatype_size*l_soa_width*l_n_chunksize);

  /* advance B prefetch pointer */
  if ( (i_xgemm_desc->prefetch & LIBXSMM_PREFETCH_BL2_VIA_C) > 0 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_gp_reg_mapping.gp_reg_b_prefetch,
                                       (l_micro_kernel_config.datatype_size*l_soa_width*i_xgemm_desc->ldb*i_xgemm_desc->m)-(l_micro_kernel_config.datatype_size*l_soa_width*l_n_chunksize));
  }

  /* adjust C pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_gp_reg_mapping.gp_reg_c,
                                     (l_micro_kernel_config.datatype_size*l_soa_width*i_xgemm_desc->ldc*i_xgemm_desc->m)-(l_micro_kernel_config.datatype_size*l_soa_width*l_n_chunksize));


  /* N loop jump back */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_nloop, i_xgemm_desc->n );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

  /* handle remainder of N loop */
  if ( l_n_remain != 0 ) {
    libxsmm_generator_spgemm_csr_asparse_soa_m_loop_avx512( io_generated_code, i_xgemm_desc, &l_loop_label_tracker, &l_micro_kernel_config, &l_gp_reg_mapping, i_arch,
                                                              i_row_idx, i_column_idx, i_values,
                                                              l_soa_width, l_gen_m_trips, l_a_is_dense, i_xgemm_desc->n - (l_n_chunksize * (l_n_chunks - 1)) );
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_generator_spgemm_csr_asparse_soa_m_loop_avx512( libxsmm_generated_code*            io_generated_code,
                                                             const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                             libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                             const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                             const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                             const char*                        i_arch,
                                                             const unsigned int*                i_row_idx,
                                                             const unsigned int*                i_column_idx,
                                                             const double*                      i_values,
                                                             const unsigned int                 i_soa_width,
                                                             const unsigned int                 i_gen_m_trips,
                                                             const unsigned int                 i_a_is_dense,
                                                             const unsigned int                 i_num_c_cols ) {
  unsigned int l_m;
  unsigned int l_n;
  unsigned int l_z;
  unsigned int l_row_elements;
  unsigned int l_b_offset;
  unsigned int l_b_total_offset;

  LIBXSMM_UNUSED(i_values);

  /* do sparse times dense soa multiplication */
  for ( l_m = 0; l_m < i_gen_m_trips; l_m++ ) {
    /* handle b offset */
    l_b_offset = 0;
    l_b_total_offset = 0;

    /* generate M loop */
    if (i_a_is_dense != 0 ) {
      libxsmm_x86_instruction_register_jump_label( io_generated_code, io_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_mloop, 1 );
    }

    l_row_elements = i_row_idx[l_m+1] - i_row_idx[l_m];
    if (l_row_elements > 0) {
      /* load C accumulator */
      for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
        if ( i_xgemm_desc->beta == 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_micro_kernel_config->instruction_set,
                                                   i_micro_kernel_config->vxor_instruction,
                                                   i_micro_kernel_config->vector_name,
                                                   l_n, l_n, l_n );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->c_vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_n*i_soa_width*i_micro_kernel_config->datatype_size,
                                            i_micro_kernel_config->vector_name,
                                            l_n, 0, 0 );
        }
        if ( (i_xgemm_desc->prefetch & LIBXSMM_PREFETCH_BL2_VIA_C) > 0 ) {
          libxsmm_x86_instruction_prefetch( io_generated_code,
                                            i_micro_kernel_config->prefetch_instruction,
                                            i_gp_reg_mapping->gp_reg_b_prefetch,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_n*i_soa_width*i_micro_kernel_config->datatype_size );
        }
      }
      /* loop over the non-zeros in A row m */
      for ( l_z = 0; l_z < l_row_elements; l_z++ ) {
        /* broadcast values of A */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->a_vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (i_row_idx[l_m] + l_z) * i_micro_kernel_config->datatype_size,
                                          i_micro_kernel_config->vector_name,
                                          i_num_c_cols, 0, 0 );
        /* multiply with B */
        for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
          l_b_offset = ((i_column_idx[i_row_idx[l_m] + l_z]*i_micro_kernel_config->datatype_size*i_soa_width*i_xgemm_desc->ldb)
                                                     +(l_n*i_soa_width*i_micro_kernel_config->datatype_size))-l_b_total_offset;
          if (l_b_offset >= 8192) {
            l_b_total_offset += l_b_offset;
            libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b,
                                             l_b_offset);
            l_b_offset = 0;
          }

          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                   i_micro_kernel_config->instruction_set,
                                                   i_micro_kernel_config->vmul_instruction,
                                                   0,
                                                   i_gp_reg_mapping->gp_reg_b,
                                                   LIBXSMM_X86_GP_REG_UNDEF,
                                                   0,
                                                   l_b_offset,
                                                   i_micro_kernel_config->vector_name,
                                                   i_num_c_cols,
                                                   l_n );
        }
      }
      /* store C accumulator */
      for ( l_n = 0; l_n < i_num_c_cols; l_n++ ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->c_vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_n*i_soa_width*i_micro_kernel_config->datatype_size,
                                          i_micro_kernel_config->vector_name,
                                          l_n, 0, 1 );
      }
    }
    /* advance C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c,
                                     i_micro_kernel_config->datatype_size*i_soa_width*i_xgemm_desc->ldc);

    /* advance B prefetch pointer */
    if ( (i_xgemm_desc->prefetch & LIBXSMM_PREFETCH_BL2_VIA_C) > 0 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_prefetch,
                                       i_micro_kernel_config->datatype_size*i_soa_width*i_xgemm_desc->ldb);
    }

    /* adjust B pointer */
    if (l_b_total_offset > 0) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_b,
                                       l_b_total_offset);
    }

    /* generate M loop */
    if (i_a_is_dense != 0 ) {
      /* advance A pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a,
                                       i_micro_kernel_config->datatype_size*i_xgemm_desc->k);

      /* M loop jump back */
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_mloop, i_xgemm_desc->m );
      libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
    }
  }

  /* reset A pointer */
  if (i_a_is_dense != 0 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a,
                                       i_micro_kernel_config->datatype_size*i_xgemm_desc->k*i_xgemm_desc->m);
  }
}

