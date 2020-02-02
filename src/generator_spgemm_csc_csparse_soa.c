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
#include "generator_spgemm_csc_csparse_soa.h"
#include "generator_gemm_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
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
void libxsmm_generator_spgemm_csc_csparse_soa( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                               const char*                     i_arch,
                                               const unsigned int*             i_row_idx,
                                               const unsigned int*             i_column_idx,
                                               const void*                     i_values ) {
  if ( strcmp(i_arch, "knl") == 0 ||
       strcmp(i_arch, "knm") == 0 ||
       strcmp(i_arch, "skx") == 0 ||
       strcmp(i_arch, "clx") == 0 ||
       strcmp(i_arch, "cpx") == 0 ) {
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
    } else {
      /* cannot happen */
    }

    libxsmm_generator_spgemm_csc_csparse_soa_avx256_512( io_generated_code,
                                                         i_xgemm_desc,
                                                         i_row_idx,
                                                         i_column_idx,
                                                         i_values );
  } else {
    fprintf( stderr, "CSC + SOA is only available for AVX/AVX2/AVX512 at this point\n" );
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_csparse_soa_avx256_512( libxsmm_generated_code*         io_generated_code,
                                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                          const unsigned int*             i_row_idx,
                                                          const unsigned int*             i_column_idx,
                                                          const void*                     i_values ) {
  unsigned int l_n = 0;
  unsigned int l_m = 0;
  unsigned int l_soa_width = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  LIBXSMM_UNUSED(i_values);

  /* select soa width */
  if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      l_soa_width = 16;
    } else {
      l_soa_width = 8;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  /* @TODO: we need to check this... however LIBXSMM descriptor setup disables A^T hard */
#if 0
  /* we need to have the A^T flag set */
  if ( (i_xgemm_desc->flags & LIBXSMM_GEMM_FLAG_TRANS_A) == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_INVALID_GEMM_CONFIG );
    return;
  }
#endif

  /*define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* matching calling convention on Linux */
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
  /* TODO: full support for Windows calling convention */
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
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

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc, 0 );

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );

  /* loop over the sparse elements of C */
  for ( l_n = 0; l_n < (unsigned int)i_xgemm_desc->n; l_n++ ) {
    unsigned int l_col_elements = i_column_idx[l_n+1] - i_column_idx[l_n];
    for ( l_m = 0; l_m < l_col_elements; ++l_m ) {
      /* set c accumulator to 0 */
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               l_micro_kernel_config.vxor_instruction,
                                               l_micro_kernel_config.vector_name,
                                               31, 31, 31 );

      /* k loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_kloop, 0 );
      libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_kloop, 1 );

      /* load a */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        l_micro_kernel_config.instruction_set,
                                        LIBXSMM_X86_INSTR_VMOVUPS,
                                        l_gp_reg_mapping.gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_micro_kernel_config.datatype_size*l_soa_width*i_row_idx[i_column_idx[l_n]+l_m],
                                        l_micro_kernel_config.vector_name,
                                        0, 0, 1, 0 );

      /* load b */
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        l_micro_kernel_config.instruction_set,
                                        LIBXSMM_X86_INSTR_VMOVUPS,
                                        l_gp_reg_mapping.gp_reg_b,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_micro_kernel_config.datatype_size*l_soa_width*l_n,
                                        l_micro_kernel_config.vector_name,
                                        1, 0, 1, 0 );

      /* FMA */
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               LIBXSMM_X86_INSTR_VFMADD231PS,
                                               l_micro_kernel_config.vector_name,
                                               0, 1, 31 );

      /* advance a and b pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_a, l_micro_kernel_config.datatype_size*l_soa_width*i_xgemm_desc->lda );
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_b, l_micro_kernel_config.datatype_size*l_soa_width*i_xgemm_desc->ldb );

      /* close k loop */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_kloop, i_xgemm_desc->k );
      libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

      /* re-set a and b pointer */
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_gp_reg_mapping.gp_reg_a, i_xgemm_desc->k*l_micro_kernel_config.datatype_size*l_soa_width*i_xgemm_desc->lda );
      libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_sub_instruction, l_gp_reg_mapping.gp_reg_b, i_xgemm_desc->k*l_micro_kernel_config.datatype_size*l_soa_width*i_xgemm_desc->ldb );

      /* reduce C */
      /* zmm31; 0000 0000 0000 0000 -> ---- ---- 0000 0000 */
      libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               l_micro_kernel_config.vector_name,
                                               31, 31, 0, 0x4e );

      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               l_micro_kernel_config.vector_name,
                                               31, 0, 31 );

      /* zmm31: ---- ---- 0000 0000 -> ---- ---- ---- 0000 */
      libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               l_micro_kernel_config.vector_name,
                                               31, 31, 0, 0xb1 );

      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               l_micro_kernel_config.vector_name,
                                               31, 0, 15 );

      /* ymm15;           ---- 0000 ->           ---- --00 */
      libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFPS,
                                               'y',
                                               15, 15, 0, 0x4e );

      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               'y',
                                               15, 0, 15 );

      /* ymm15;           ---- --00 ->           ---- ---0 */
      libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFPS,
                                               'y',
                                               15, 15, 0, 0x1 );

      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               l_micro_kernel_config.instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               'y',
                                               15, 0, 15 );

      /* update sparse C */
      if ( 0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags) ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          l_micro_kernel_config.instruction_set,
                                          LIBXSMM_X86_INSTR_VMOVSS,
                                          l_gp_reg_mapping.gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_micro_kernel_config.datatype_size*(i_column_idx[l_n]+l_m),
                                          'x',
                                          0, 0, 1, 0 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 l_micro_kernel_config.instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDSS,
                                                 'x',
                                                 15, 0, 15 );
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        l_micro_kernel_config.instruction_set,
                                        LIBXSMM_X86_INSTR_VMOVSS,
                                        l_gp_reg_mapping.gp_reg_c,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_micro_kernel_config.datatype_size*(i_column_idx[l_n]+l_m),
                                        'x',
                                        15, 0, 1, 1 );
    }
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );
}

