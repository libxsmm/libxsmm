/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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
#include "generator_gemm_rm_ac_soa.h"
#include "generator_gemm_common.h"
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

LIBXSMM_API void libxsmm_generator_gemm_rm_ac_soa( libxsmm_generated_code*         io_generated_code,
                                                   const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                   const char*                     i_arch ) {
  if ( strcmp(i_arch, "knl") == 0 ||
       strcmp(i_arch, "knm") == 0 ||
       strcmp(i_arch, "skx") == 0 ||
       strcmp(i_arch, "hsw") == 0 ||
       strcmp(i_arch, "snb") == 0 ) {
    libxsmm_generator_gemm_rm_ac_soa_avx256_512( io_generated_code,
                                                         i_xgemm_desc,
                                                         i_arch );
  } else {
    fprintf( stderr, "RM AC SOA is only available for AVX/AVX2/AVX512 at this point\n" );
    exit(-1);
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_rm_ac_soa_avx256_512( libxsmm_generated_code*         io_generated_code,
                                                                     const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                     const char*                     i_arch ) {
  unsigned int l_soa_width = 0;
  unsigned int l_max_reg_block = 0;
  unsigned int l_n1_range = 0;
  unsigned int l_n2_range = 0;
  unsigned int l_n1_block = 0;
  unsigned int l_n2_block = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* select soa width */
  if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    if ( strcmp(i_arch, "knl") == 0 ||
         strcmp(i_arch, "knm") == 0 ||
         strcmp(i_arch, "icl") == 0 ||
         strcmp(i_arch, "skx") == 0 ) {
      l_soa_width = 8;
      l_max_reg_block = 28;
    } else {
      l_soa_width = 4;
      l_max_reg_block = 14;
    }
  } else {
    if ( strcmp(i_arch, "knl") == 0 ||
         strcmp(i_arch, "knm") == 0 ||
         strcmp(i_arch, "icl") == 0 ||
         strcmp(i_arch, "skx") == 0 ) {
      l_soa_width = 16;
      l_max_reg_block = 28;
    } else {
      l_soa_width = 8;
      l_max_reg_block = 14;
    }
  }

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* matching calling convention on Linux */
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
  /* TODO: full support for Windows calling convention */
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_RSI;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
#endif
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

  /* calculate the chunk size of current columns to work on */
  if ( libxsmm_compute_equalized_blocking( i_xgemm_desc->n, l_max_reg_block, &l_n1_range, &l_n1_block, &l_n2_range, &l_n2_block ) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );

  /* m loop */
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_mloop, 1 );

  /* loop over n-blocks */
  if ( l_n1_block == i_xgemm_desc->n ) {
    /* no N loop at all */
    libxsmm_generator_gemm_rm_ac_soa_avx256_512_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                         i_arch, l_soa_width, i_xgemm_desc->n );
  } else if ( (l_n1_range > 0) && (l_n2_range > 0) ) {
    /* reset n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_nloop, 0 );

    /* we have two ranges */
    /* first range */
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n1_block );

    libxsmm_generator_gemm_rm_ac_soa_avx256_512_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                         i_arch, l_soa_width, l_n1_block );

    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n1_range );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

    /* second range */
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n2_block );

    libxsmm_generator_gemm_rm_ac_soa_avx256_512_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                         i_arch, l_soa_width, l_n2_block );

    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_nloop, i_xgemm_desc->n );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

    /* reset B pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     l_micro_kernel_config.alu_sub_instruction,
                                     l_gp_reg_mapping.gp_reg_b,
                                     i_xgemm_desc->n * l_micro_kernel_config.datatype_size );

    /* reset C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     l_micro_kernel_config.alu_sub_instruction,
                                     l_gp_reg_mapping.gp_reg_c,
                                     i_xgemm_desc->n * l_soa_width * l_micro_kernel_config.datatype_size );
  } else if ( (l_n1_range > 0) && (l_n2_range == 0) ) {
    /* reset n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_nloop, 0 );

    /* we have one range */
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n1_block );

    libxsmm_generator_gemm_rm_ac_soa_avx256_512_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                         i_arch, l_soa_width, l_n1_block );

    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_nloop, i_xgemm_desc->n );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

    /* reset B pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     l_micro_kernel_config.alu_sub_instruction,
                                     l_gp_reg_mapping.gp_reg_b,
                                     i_xgemm_desc->n * l_micro_kernel_config.datatype_size );

    /* reset C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     l_micro_kernel_config.alu_sub_instruction,
                                     l_gp_reg_mapping.gp_reg_c,
                                     i_xgemm_desc->n * l_soa_width * l_micro_kernel_config.datatype_size );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* advance A pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_a,
                                   l_micro_kernel_config.datatype_size*l_soa_width*i_xgemm_desc->lda);

  /* advance C pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_c,
                                   l_micro_kernel_config.datatype_size*l_soa_width*i_xgemm_desc->ldc);

  /* close m loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );
}

LIBXSMM_API_INTERN void libxsmm_generator_gemm_rm_ac_soa_avx256_512_kloop( libxsmm_generated_code*            io_generated_code,
                                                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                           const char*                        i_arch,
                                                                           const unsigned int                 i_soa_width,
                                                                           const unsigned int                 i_n_blocking ) {
  unsigned int l_n = 0;
  unsigned int l_lcl_k = 0;

  /* load C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
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
                                        l_n, 0, 1, 0 );
    }
  }

  /* k loop */
  libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, 0, 1 );

  /* full vector load of A */
  /* prepare KNM's QMADD */
  for ( l_lcl_k = 0; l_lcl_k < 1; l_lcl_k++ ) {
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->a_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_a,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_lcl_k*i_soa_width*i_micro_kernel_config->datatype_size,
                                      i_micro_kernel_config->vector_name,
                                      i_n_blocking+l_lcl_k, 0, 1, 0 );
  }

  /* loop over the register block */
  for ( l_n = 0; l_n < i_n_blocking; ++l_n ) {
    if ( strcmp(i_arch, "knl") == 0 ||
         strcmp(i_arch, "knm") == 0 ||
         strcmp(i_arch, "icl") == 0 ||
         strcmp(i_arch, "skx") == 0 ) {
      libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->vmul_instruction,
                                               1,
                                               i_gp_reg_mapping->gp_reg_b,
                                               LIBXSMM_X86_GP_REG_UNDEF,
                                               0,
                                               l_n * i_micro_kernel_config->datatype_size,
                                               i_micro_kernel_config->vector_name,
                                               i_n_blocking,
                                               l_n );
    } else if ( strcmp(i_arch, "hsw") == 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->b_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_b,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_n * i_micro_kernel_config->datatype_size,
                                        i_micro_kernel_config->vector_name,
                                        i_n_blocking+1, 0, 1, 0 );
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->vmul_instruction,
                                               i_micro_kernel_config->vector_name,
                                               i_n_blocking,
                                               i_n_blocking+1,
                                               l_n );
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->b_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_b,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_n * i_micro_kernel_config->datatype_size,
                                        i_micro_kernel_config->vector_name,
                                        i_n_blocking+1, 0, 1, 0 );
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->vmul_instruction,
                                               i_micro_kernel_config->vector_name,
                                               i_n_blocking,
                                               i_n_blocking+1,
                                               i_n_blocking+1 );
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->vadd_instruction,
                                               i_micro_kernel_config->vector_name,
                                               i_n_blocking+1,
                                               l_n,
                                               l_n );
    }
  }

  /* advance A pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_a,
                                   i_soa_width * i_micro_kernel_config->datatype_size );

  /* advance B pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_b,
                                   i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size );

  /* close k loop */
  libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, 0, i_xgemm_desc->k, 0 );

  /* store C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      i_micro_kernel_config->c_vmove_instruction,
                                      i_gp_reg_mapping->gp_reg_c,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      l_n*i_soa_width*i_micro_kernel_config->datatype_size,
                                      i_micro_kernel_config->vector_name,
                                      l_n, 0, 0, 1 );
  }

  /* reset A pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_sub_instruction,
                                   i_gp_reg_mapping->gp_reg_a,
                                   i_xgemm_desc->k * i_soa_width * i_micro_kernel_config->datatype_size );

  /* fix pointers */
  if ( i_xgemm_desc->n != i_n_blocking ) {
    /* advance B pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_sub_instruction,
                                     i_gp_reg_mapping->gp_reg_b,
                                     (i_xgemm_desc->k * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size) - (i_n_blocking * i_micro_kernel_config->datatype_size) );

    /* advance C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_c,
                                     i_n_blocking * i_soa_width * i_micro_kernel_config->datatype_size );
  } else {
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_sub_instruction,
                                     i_gp_reg_mapping->gp_reg_b,
                                     i_xgemm_desc->k * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size );
  }
}

