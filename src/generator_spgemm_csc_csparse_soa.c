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
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_csparse_soa_axv256_512_single( libxsmm_generated_code*            io_generated_code,
                                                                 libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                 const unsigned int*                i_row_idx,
                                                                 const unsigned int*                i_column_idx,
                                                                 const unsigned int                 i_packed_width,
                                                                 const unsigned int                 i_n,
                                                                 const unsigned int                 i_m ) {
  /* compute packed loop trip count */
#if 0
  unsigned int l_simd_packed_remainder = 0;
#endif
  unsigned int l_simd_packed_iters = 0;
  unsigned int l_simd_packed_width = 0;

  /* select simd packing width */
  if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      l_simd_packed_width = 8;
    } else {
      l_simd_packed_width = 4;
    }
  } else {
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      l_simd_packed_width = 16;
    } else {
      l_simd_packed_width = 8;
    }
  }
#if 0
  l_simd_packed_remainder = i_packed_width % l_simd_packed_width;
#endif
  l_simd_packed_iters = i_packed_width/l_simd_packed_width;

  /* set c accumulator to 0 */
  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           i_micro_kernel_config->vxor_instruction,
                                           i_micro_kernel_config->vector_name,
                                           31, 31, 31 );

  /* k loop header */
  if ( i_xgemm_desc->k > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_kloop, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_kloop, 1 );
  }

  /* packed loop header */
  if ( l_simd_packed_iters > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_1, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, 1 );
  }

  /* load b */
  libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    LIBXSMM_X86_INSTR_VMOVUPS,
                                    i_gp_reg_mapping->gp_reg_b,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_micro_kernel_config->datatype_size*i_packed_width*i_n,
                                    i_micro_kernel_config->vector_name,
                                    0, 0, 1, 0 );

  /* FMA with fused load of a */
  libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VFMADD231PS,
                                           0,
                                           i_gp_reg_mapping->gp_reg_a,
                                           LIBXSMM_X86_GP_REG_UNDEF, 0,
                                           i_micro_kernel_config->datatype_size*i_packed_width*i_row_idx[i_column_idx[i_n]+i_m],
                                           i_micro_kernel_config->vector_name,
                                           0,
                                           31 );

  /* packed loop footer */
  if ( l_simd_packed_iters > 1 ) {
    /* advance a and b pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, i_micro_kernel_config->datatype_size*l_simd_packed_width );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, i_micro_kernel_config->datatype_size*l_simd_packed_width );

    /* check loop bound */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_help_1, l_simd_packed_iters );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

    /* re-set a and b pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, l_simd_packed_iters*i_micro_kernel_config->datatype_size*l_simd_packed_width );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_b, l_simd_packed_iters*i_micro_kernel_config->datatype_size*l_simd_packed_width );
  }

  /* k loop footer */
  if ( i_xgemm_desc->k > 1 ) {
    /* advance a and b pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, i_micro_kernel_config->datatype_size*i_packed_width*i_xgemm_desc->lda );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, i_micro_kernel_config->datatype_size*i_packed_width*i_xgemm_desc->ldb );

    /* close k loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_kloop, i_xgemm_desc->k );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

    /* re-set a and b pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, i_xgemm_desc->k*i_micro_kernel_config->datatype_size*i_packed_width*i_xgemm_desc->lda );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_b, i_xgemm_desc->k*i_micro_kernel_config->datatype_size*i_packed_width*i_xgemm_desc->ldb );
  }

  /* reduce C */
  /* zmm31; 0000 0000 0000 0000 -> ---- ---- 0000 0000 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           31, 31, 0, 0x4e );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           31, 0, 31 );

  /* zmm31: ---- ---- 0000 0000 -> ---- ---- ---- 0000 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           31, 31, 0, 0xb1 );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           31, 0, 15 );

  /* ymm15;           ---- 0000 ->           ---- --00 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           'y',
                                           15, 15, 0, 0x4e );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           'y',
                                           15, 0, 15 );

  /* ymm15;           ---- --00 ->           ---- ---0 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           'y',
                                           15, 15, 0, 0x1 );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           'y',
                                           15, 0, 15 );

  /* update sparse C */
  if ( 0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags) ) {
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      LIBXSMM_X86_INSTR_VMOVSS,
                                      i_gp_reg_mapping->gp_reg_c,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      i_micro_kernel_config->datatype_size*(i_column_idx[i_n]+i_m),
                                      'x',
                                      0, 0, 1, 0 );

    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDSS,
                                             'x',
                                             15, 0, 15 );
  }

  libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    LIBXSMM_X86_INSTR_VMOVSS,
                                    i_gp_reg_mapping->gp_reg_c,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_micro_kernel_config->datatype_size*(i_column_idx[i_n]+i_m),
                                    'x',
                                    15, 0, 1, 1 );
}


LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_csparse_soa_axv256_512_16accs( libxsmm_generated_code*            io_generated_code,
                                                                 libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                 const unsigned int*                i_row_idx,
                                                                 const unsigned int*                i_column_idx,
                                                                 const unsigned int                 i_packed_width,
                                                                 const unsigned int                 i_n,
                                                                 const unsigned int                 i_m,
                                                                 const unsigned int                 i_m_blocking ) {
  /* some helper variables */
  unsigned int l_i, l_max_m, l_mask_reg, l_mask_val;
  /* compute packed loop trip count */
#if 0
  unsigned int l_simd_packed_remainder = 0;
#endif
  unsigned int l_simd_packed_iters = 0;
  unsigned int l_simd_packed_width = 0;

  /* select simd packing width */
  if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      l_simd_packed_width = 8;
    } else {
      l_simd_packed_width = 4;
    }
  } else {
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      l_simd_packed_width = 16;
    } else {
      l_simd_packed_width = 8;
    }
  }
#if 0
  l_simd_packed_remainder = i_packed_width % l_simd_packed_width;
#endif
  l_simd_packed_iters = i_packed_width/l_simd_packed_width;

  /* we only generated for AVX512 for now, max m is 16; max_m is used for init and reduction */
  l_max_m = 16;
  l_mask_reg = 1;

  /* load maske register */
  l_mask_val = 0xffff >> (16-i_m_blocking);
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_help_0, l_mask_val );
  libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVW, i_gp_reg_mapping->gp_reg_help_0, l_mask_reg );

  /* set c accumulator to 0 */
  for ( l_i = 0; l_i < l_max_m; ++l_i ) {
    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             i_micro_kernel_config->vxor_instruction,
                                             i_micro_kernel_config->vector_name,
                                             l_i, l_i, l_i );
  }

  /* k loop header */
  if ( i_xgemm_desc->k > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_kloop, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_kloop, 1 );
  }

  /* packed loop header */
  if ( l_simd_packed_iters > 1 ) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_1, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_1, 1 );
  }

  /* load b */
  libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    LIBXSMM_X86_INSTR_VMOVUPS,
                                    i_gp_reg_mapping->gp_reg_b,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_micro_kernel_config->datatype_size*i_packed_width*i_n,
                                    i_micro_kernel_config->vector_name,
                                    31, 0, 1, 0 );

  /* FMA with fused load of a */
  for ( l_i = i_m; l_i < (i_m + i_m_blocking); ++l_i ) {
    libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VFMADD231PS,
                                             0,
                                             i_gp_reg_mapping->gp_reg_a,
                                             LIBXSMM_X86_GP_REG_UNDEF, 0,
                                             i_micro_kernel_config->datatype_size*i_packed_width*i_row_idx[i_column_idx[i_n]+l_i],
                                             i_micro_kernel_config->vector_name,
                                             31,
                                             l_i%16 );
  }

  /* packed loop footer */
  if ( l_simd_packed_iters > 1 ) {
    /* advance a and b pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, i_micro_kernel_config->datatype_size*l_simd_packed_width );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, i_micro_kernel_config->datatype_size*l_simd_packed_width );

    /* check loop bound */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_help_1, l_simd_packed_iters );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

    /* re-set a and b pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, l_simd_packed_iters*i_micro_kernel_config->datatype_size*l_simd_packed_width );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_b, l_simd_packed_iters*i_micro_kernel_config->datatype_size*l_simd_packed_width );
  }

  /* k loop footer */
  if ( i_xgemm_desc->k > 1 ) {
    /* advance a and b pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_a, i_micro_kernel_config->datatype_size*i_packed_width*i_xgemm_desc->lda );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b, i_micro_kernel_config->datatype_size*i_packed_width*i_xgemm_desc->ldb );

    /* close k loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_kloop, i_xgemm_desc->k );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

    /* re-set a and b pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, i_xgemm_desc->k*i_micro_kernel_config->datatype_size*i_packed_width*i_xgemm_desc->lda );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_b, i_xgemm_desc->k*i_micro_kernel_config->datatype_size*i_packed_width*i_xgemm_desc->ldb );
  }

  /* reduce C */
  /* 1st stage */
  /* zmm0/zmm4; 4444 4444 4444 4444 / 0000 0000 0000 0000 -> zmm0: 4444 4444 0000 0000 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           4, 0, 16, 0x44 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           4, 0, 17, 0xee );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 0 );

  if ( i_m_blocking > 7 ) {
    /* zmm8/zmm12; cccc cccc cccc cccc / 8888 8888 8888 8888 -> zmm8: cccc cccc 8888 8888 */
    libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             12, 8, 16, 0x44 );

    libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             12, 8, 17, 0xee );

    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 8 );
  }

  /* zmm1/zmm5; 5555 5555 5555 5555 / 1111 1111 1111 1111 -> zmm1: 5555 5555 1111 1111 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           5, 1, 16, 0x44 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           5, 1, 17, 0xee );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 1 );

  if ( i_m_blocking > 8 ) {
    /* zmm9/zmm13; dddd dddd dddd dddd / 9999 9999 9999 9999 -> zmm9: dddd dddd 9999 9999 */
    libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             13, 9, 16, 0x44 );

    libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             13, 9, 17, 0xee );

    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 9 );
  }

  /* zmm2/zmm6; 6666 6666 6666 6666 / 2222 2222 2222 2222 -> zmm2: 6666 6666 2222 2222 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           6, 2, 16, 0x44 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           6, 2, 17, 0xee );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 2 );

  if ( i_m_blocking > 9 ) {
    /* zmm10/zmm14; eeee eeee eeee eeee / aaaa aaaa aaaa aaaa -> zmm10: eeee eeee aaaa aaaa */
    libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             14, 10, 16, 0x44 );

    libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             14, 10, 17, 0xee );

    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 10 );
  }

  /* zmm3/zmm7; 7777 7777 7777 7777 / 3333 3333 3333 3333  -> zmm3: 7777 7777 3333 3333 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           7, 3, 16, 0x44 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           7, 3, 17, 0xee );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 3 );

  if ( i_m_blocking > 10 ) {
    /* zmm11/zmm15; ffff ffff ffff ffff / bbbb bbbb bbbb bbbb  -> zmm11: ffff ffff bbbb bbbb */
    libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             15, 11, 16, 0x44 );

    libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             15, 11, 17, 0xee );

    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 11 );
  }

  /* 2nd stage */
  /* zmm0/zmm8; 4444 4444 0000 0000 / cccc cccc 8888 8888  -> zmm0: cccc 8888 4444 0000 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           8, 0, 16, 0x88 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           8, 0, 17, 0xdd );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 0 );

  /* zmm1/zmm9; 5555 5555 1111 1111 / dddd dddd 9999 9999  -> zmm1: dddd 9999 5555 1111 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           9, 1, 16, 0x88 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           9, 1, 17, 0xdd );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 1 );

  /* zmm2/zmm10; 6666 6666 2222 2222 / eeee eeee aaaa aaaa  -> zmm2: eeee aaaa 6666 2222 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           10, 2, 16, 0x88 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           10, 2, 17, 0xdd );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 2 );

  /* zmm3/zmm11:  7777 7777 3333 3333 / ffff ffff bbbb bbbb  -> zmm3: ffff bbbb 7777 3333 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           11, 3, 16, 0x88 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           11, 3, 17, 0xdd );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 3 );

  /* 3rd stage */
  /* zmm0/zmm1; cccc 8888 4444 0000 / dddd 9999 5555 1111  -> zmm0: ddcc 9988 5544 1100 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           i_micro_kernel_config->vector_name,
                                           1, 0, 16, 0x44 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           i_micro_kernel_config->vector_name,
                                           1, 0, 17, 0xee );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 0 );

  /* zmm2/zmm3; eeee aaaa 6666 2222 / ffff bbbb 7777 3333  -> zmm2: ffee bbaa 7766 3322 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           i_micro_kernel_config->vector_name,
                                           3, 2, 16, 0x44 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           i_micro_kernel_config->vector_name,
                                           3, 2, 17, 0xee );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 2 );

  /* 4th stage */
  /* zmm0/zmm2; ddcc 9988 5544 1100 / ffee bbaa 7766 3322  -> zmm0: fedc ba98 7654 3210 */
  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           i_micro_kernel_config->vector_name,
                                           2, 0, 16, 0x88 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           i_micro_kernel_config->vector_name,
                                           2, 0, 17, 0xdd );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           16, 17, 0 );

  /* update sparse C */
  if ( 0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags) ) {
    libxsmm_x86_instruction_vec_move( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      LIBXSMM_X86_INSTR_VMOVUPS,
                                      i_gp_reg_mapping->gp_reg_c,
                                      LIBXSMM_X86_GP_REG_UNDEF, 0,
                                      i_micro_kernel_config->datatype_size*(i_column_idx[i_n]+i_m),
                                      i_micro_kernel_config->vector_name,
                                      1, l_mask_reg, 1, 0 );

    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             0, 1, 1 );
  }

  libxsmm_x86_instruction_vec_move( io_generated_code,
                                    i_micro_kernel_config->instruction_set,
                                    LIBXSMM_X86_INSTR_VMOVUPS,
                                    i_gp_reg_mapping->gp_reg_c,
                                    LIBXSMM_X86_GP_REG_UNDEF, 0,
                                    i_micro_kernel_config->datatype_size*(i_column_idx[i_n]+i_m),
                                    i_micro_kernel_config->vector_name,
                                    1, l_mask_reg, 0, 1 );
}


LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_csparse_soa( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                               const char*                     i_arch,
                                               const unsigned int*             i_row_idx,
                                               const unsigned int*             i_column_idx,
                                               const void*                     i_values,
                                               const unsigned int              i_packed_width ) {
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
                                                         i_values,
                                                         i_packed_width );
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
                                                          const void*                     i_values,
                                                          const unsigned int              i_packed_width ) {
  unsigned int l_n = 0;
  unsigned int l_m = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  LIBXSMM_UNUSED(i_values);

  /* select soa width */
  if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      if ( i_packed_width % 16 != 0 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
        return;
      }
    } else {
      if ( i_packed_width % 8 != 0 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
        return;
      }
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
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R15;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_R11;
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

    if ( l_col_elements > 2 ) {
      for ( l_m = 0; l_m < (l_col_elements/16)*16; l_m+=16 ) {
        libxsmm_generator_spgemm_csc_csparse_soa_axv256_512_16accs( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                                    i_row_idx, i_column_idx, i_packed_width, l_n, l_m, 16 );
      }
      if ( l_col_elements % 16 != 0 ) {
        libxsmm_generator_spgemm_csc_csparse_soa_axv256_512_16accs( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                                    i_row_idx, i_column_idx, i_packed_width, l_n, l_m, l_col_elements%16 );
      }
    } else {
      for ( l_m = 0; l_m < l_col_elements; ++l_m ) {
        libxsmm_generator_spgemm_csc_csparse_soa_axv256_512_single( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                                    i_row_idx, i_column_idx, i_packed_width, l_n, l_m );
      }
    }
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );
}

