/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_packed_gemm_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common_aarch64.h"
#include "generator_gemm_common_aarch64.h"

#define LIBXSMM_GENERATOR_PACKED_GEMM_AARCH64_MASK_REG LIBXSMM_AARCH64_SVE_REG_P1

LIBXSMM_API_INTERN
void libxsmm_generator_packed_gemm_aarch64( libxsmm_generated_code*         io_generated_code,
                                            const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                            const unsigned int              i_packed_width ) {
  unsigned int l_max_n_reg_block = 0;
  unsigned int l_m_block = 0;
  unsigned int l_m_remainder = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* select accumulator blocking */
  /* TODO: we could do more aggressive blockings if needed */
  l_max_n_reg_block = 28;

  /* define gp register mapping */
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_AARCH64_GP_REG_X4;
  /*l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_AARCH64_GP_REG_X5;*/
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_AARCH64_GP_REG_X7;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_AARCH64_GP_REG_X8;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_AARCH64_GP_REG_X9;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_AARCH64_GP_REG_X10;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_AARCH64_GP_REG_X11;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_AARCH64_GP_REG_X12;     /* this is the SIMD packed register loop */
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_AARCH64_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_AARCH64_GP_REG_UNDEF;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_aarch64( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc );

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xf );

  /* implementing load from struct */
  if ( ((LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI & i_xgemm_desc->flags) == LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI) ) {
    /* RDI holds the pointer to the struct, so lets first move this one into R15 */
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR,
                                                         l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_param_struct, l_gp_reg_mapping.gp_reg_help_1,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    /* A pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 32, l_gp_reg_mapping.gp_reg_a );
    /* B pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 64, l_gp_reg_mapping.gp_reg_b );
    /* C pointer */
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                     l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 96, l_gp_reg_mapping.gp_reg_c );
    if ( i_xgemm_desc->prefetch != LIBXSMM_GEMM_PREFETCH_NONE ) {
      /* A prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 56, l_gp_reg_mapping.gp_reg_a_prefetch );
      /* B prefetch pointer */
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF,
                                       l_gp_reg_mapping.gp_reg_help_1, LIBXSMM_AARCH64_GP_REG_UNDEF, 88, l_gp_reg_mapping.gp_reg_b_prefetch );
    }
  } else {
#if 0
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_ABI );
    return;
#endif
  }

  /* set P0 in case of SVE */
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_AARCH64_SVE_REG_P0,
                                                  -1,
                                                  l_gp_reg_mapping.gp_reg_help_0 );
  }

  /* some very simple m-blocking */
  if ( i_xgemm_desc->m >= 4 ) {
    l_m_block = 4;
    l_m_remainder = i_xgemm_desc->m % 4;
    l_max_n_reg_block = 6;
  } else {
    l_m_block = 0;
    l_m_remainder = i_xgemm_desc->m;
  }

  if ( l_m_block == 4 ) {
    /* m loop */
    libxsmm_generator_loop_header_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m-l_m_remainder );

    /* m loop body */
    libxsmm_generator_packed_gemm_aarch64_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                                 i_xgemm_desc, i_packed_width, l_max_n_reg_block, l_m_block );

    /* close m loop */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, &l_loop_label_tracker, l_gp_reg_mapping.gp_reg_mloop, 4 );
  }

  if ( l_m_remainder != 0 ) {
    if ( l_m_remainder == 3 ) {
      l_m_block = 3;
      l_max_n_reg_block = 8;
    } else if ( l_m_remainder == 2 ) {
      l_m_block = 2;
      l_max_n_reg_block = 12;
    } else if ( l_m_remainder == 1 ) {
      l_m_block = 1;
      l_max_n_reg_block = 28;
    }

    /* m loop body for remainder */
    libxsmm_generator_packed_gemm_aarch64_mloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config,
                                                 i_xgemm_desc, i_packed_width, l_max_n_reg_block, l_m_block );
  }

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xf );
}


LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_aarch64_mloop( libxsmm_generated_code*            io_generated_code,
                                                                     libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                     const unsigned int                 i_packed_width,
                                                                     const unsigned int                 i_max_n_reg_block,
                                                                     const unsigned int                 i_m_blocking ) {
  unsigned int l_n1_range = 0;
  unsigned int l_n2_range = 0;
  unsigned int l_n1_block = 0;
  unsigned int l_n2_block = 0;

  /* calculate the chunk size of current columns to work on */
  if ( libxsmm_compute_equalized_blocking( i_xgemm_desc->n, i_max_n_reg_block, &l_n1_range, &l_n1_block, &l_n2_range, &l_n2_block ) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* loop over n-blocks */
  if ( l_n1_block == i_xgemm_desc->n ) {
    /* no N loop at all */
    libxsmm_generator_packed_gemm_aarch64_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                 i_packed_width, i_xgemm_desc->n, i_m_blocking );
  } else if ( (l_n1_range > 0) && (l_n2_range > 0) ) {
    /* we have two ranges */
    /* first range */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_nloop, l_n1_range );

    libxsmm_generator_packed_gemm_aarch64_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                 i_packed_width, l_n1_block, i_m_blocking );

    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_nloop, l_n1_block );

    /* second range */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_nloop, i_xgemm_desc->n - l_n1_range );

    libxsmm_generator_packed_gemm_aarch64_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                 i_packed_width, l_n2_block, i_m_blocking );

    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_nloop, l_n2_block );


    /* reset B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (long long)i_xgemm_desc->n * i_packed_width * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in );

    /* reset C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_xgemm_desc->n * i_packed_width * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size_out );
  } else if ( (l_n1_range > 0) && (l_n2_range == 0) ) {
    /* reset n loop */
    /* we have only one range */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_nloop, i_xgemm_desc->n );

    libxsmm_generator_packed_gemm_aarch64_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                       i_packed_width, l_n1_block, i_m_blocking );

    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_nloop, l_n1_block );

    /* reset B pointeri */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (long long)i_xgemm_desc->n * i_packed_width * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in );

    /* reset C pointeri */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_xgemm_desc->n * i_packed_width * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size_out );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* advance A pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                 (long long)i_micro_kernel_config->datatype_size_in*i_packed_width*i_m_blocking );

  /* advance C pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                 (long long)i_micro_kernel_config->datatype_size_out*i_packed_width*i_m_blocking );
}

LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_aarch64_kloop( libxsmm_generated_code*            io_generated_code,
                                                                     libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                     const unsigned int                 i_packed_width,
                                                                     const unsigned int                 i_n_blocking,
                                                                     const unsigned int                 i_m_blocking ) {
  /* calculate how many iterations we need */
  unsigned int l_simd_packed_remainder = 0;
  unsigned int l_simd_packed_iters = 0;
  unsigned int l_simd_packed_width = 0;

  /* set function pointer based on supported SIMD arch */
  void (*l_generator_microkernel)(libxsmm_generated_code*, libxsmm_loop_label_tracker*, const libxsmm_gp_reg_mapping*,
                                  const libxsmm_micro_kernel_config*, const libxsmm_gemm_descriptor*, const unsigned int,
                                  const unsigned int, const unsigned int, const unsigned int, const unsigned int );

  /* select simd packing width and accumulator blocking */
  if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GEMM_GETENUM_AB_COMMON_PREC( i_xgemm_desc->datatype ) ) {
    if ( io_generated_code->arch >= LIBXSMM_AARCH64_SVE128 ) {
      if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE256 ) {
        l_simd_packed_width = 2;
      } else if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE512 ) {
        l_simd_packed_width = 4;
      } else {
        l_simd_packed_width = 8;
      }
    } else { /* asimd */
      l_simd_packed_width = 2;
    }
  } else {
    if ( io_generated_code->arch >= LIBXSMM_AARCH64_SVE128 ) {
      if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE256 ) {
        l_simd_packed_width = 4;
      } else if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE512 ) {
        l_simd_packed_width = 8;
      } else {
        l_simd_packed_width = 16;
      }
    } else { /* asimd */
      l_simd_packed_width = 4;
    }
  }

  l_generator_microkernel = libxsmm_generator_packed_gemm_aarch64_kloop_simd_packed;
  l_simd_packed_remainder = i_packed_width % l_simd_packed_width;
  l_simd_packed_iters = i_packed_width/l_simd_packed_width;

  /* set P1 in case of SVE for remainder handling */
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_set_p_register_aarch64_sve( io_generated_code,
                                                  LIBXSMM_GENERATOR_PACKED_GEMM_AARCH64_MASK_REG,
                                                  l_simd_packed_remainder * i_micro_kernel_config->datatype_size_in,
                                                  i_gp_reg_mapping->gp_reg_help_0 );
  }

  /* check if we have a single SIMD divisor */
  if ( l_simd_packed_width == i_packed_width ) {
    /* run inner compute kernel */
    l_generator_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                             i_packed_width, l_simd_packed_width, l_simd_packed_width, i_n_blocking, i_m_blocking );
  /* check if we have a perfect SIMD divisor */
  } else if ( l_simd_packed_remainder == 0 ) {
    /* initialize packed loop */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, i_packed_width );

    /* run inner compute kernel */
    l_generator_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                             i_packed_width, l_simd_packed_width, l_simd_packed_width, i_n_blocking, i_m_blocking );

    /* advance A pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)l_simd_packed_width * i_micro_kernel_config->datatype_size_in );

    /* advance B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (long long)l_simd_packed_width * i_micro_kernel_config->datatype_size_in );

    /* advance C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)l_simd_packed_width * i_micro_kernel_config->datatype_size_out );

    /* jump back to pack loop label */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, l_simd_packed_width );

    /* reset A pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)l_simd_packed_iters * l_simd_packed_width * i_micro_kernel_config->datatype_size_in );

    /* reset B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (long long)l_simd_packed_iters * l_simd_packed_width * i_micro_kernel_config->datatype_size_in );

    /* reset C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)l_simd_packed_iters * l_simd_packed_width * i_micro_kernel_config->datatype_size_out );
  /* we need masking and have less than SIMD width */
  } else if ( l_simd_packed_width > i_packed_width  ) {
    /* run inner compute kernel with mask */
    l_generator_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                             i_packed_width, l_simd_packed_width, l_simd_packed_remainder, i_n_blocking, i_m_blocking );
  /* we need the general case */
  } else {
    /* initialize packed loop */
    libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, i_packed_width - l_simd_packed_remainder );

    /* run inner compute kernel */
    l_generator_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                             i_packed_width, l_simd_packed_width, l_simd_packed_width, i_n_blocking, i_m_blocking );

    /* advance A pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)l_simd_packed_width * i_micro_kernel_config->datatype_size_in );

    /* advance B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (long long)l_simd_packed_width * i_micro_kernel_config->datatype_size_in );

    /* advance C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)l_simd_packed_width * i_micro_kernel_config->datatype_size_out );

    /* jump back to pack loop label */
    libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_help_3, l_simd_packed_width );

    /* run remainder kernel */
    l_generator_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                             i_packed_width, l_simd_packed_width, l_simd_packed_remainder, i_n_blocking, i_m_blocking );

    /* reset A pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)l_simd_packed_iters * l_simd_packed_width * i_micro_kernel_config->datatype_size_in );

    /* reset B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (long long)l_simd_packed_iters * l_simd_packed_width * i_micro_kernel_config->datatype_size_in );

    /* reset C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)l_simd_packed_iters * l_simd_packed_width * i_micro_kernel_config->datatype_size_out );
  }

  /* advance B and C pointers if N is bigger than our register blocking */
  if ( i_xgemm_desc->n != i_n_blocking ) {
    /* advance B pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (long long)i_n_blocking * i_packed_width * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size_in );

    /* advance C pointer */
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_n_blocking * i_packed_width * i_xgemm_desc->ldc * i_micro_kernel_config->datatype_size_out );
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_aarch64_load_C( libxsmm_generated_code*            io_generated_code,
                                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                      const unsigned int                 i_n_blocking,
                                                                      const unsigned int                 i_m_blocking,
                                                                      const unsigned int                 i_use_masking,
                                                                      const unsigned int                 i_packed_width ) {
  unsigned int l_n = 0;
  unsigned int l_m = 0;
  unsigned int l_is_sve = ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) ? 1 : 0;

  /* load C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < i_m_blocking; l_m++ ) {
      if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
        if ( l_is_sve ) {
          libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                   LIBXSMM_AARCH64_INSTR_SVE_EOR_V,
                                                   (l_n*i_m_blocking) + l_m, (l_n*i_m_blocking) + l_m, (unsigned char)-1, (l_n*i_m_blocking) + l_m,
                                                   LIBXSMM_AARCH64_SVE_REG_UNDEF,
                                                   LIBXSMM_AARCH64_SVE_TYPE_D );
        } else {
          libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,
                                                     (l_n*i_m_blocking) + l_m, (l_n*i_m_blocking) + l_m, 0, (l_n*i_m_blocking) + l_m,
                                                     LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B );
        }
      } else {
        libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2,
                                                          (l_n*i_m_blocking) + l_m, i_micro_kernel_config->datatype_size_out, i_use_masking,
                                                          0, 0, LIBXSMM_GENERATOR_PACKED_GEMM_AARCH64_MASK_REG );
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                       i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                       (long long)i_packed_width*i_micro_kernel_config->datatype_size_out );
      }
    }
    if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                     (long long)(i_xgemm_desc->ldc-i_m_blocking)*i_packed_width*i_micro_kernel_config->datatype_size_out  );

    }
  }
  /* reset C point for stores */
  if (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) {
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)i_n_blocking*i_xgemm_desc->ldc*i_packed_width*i_micro_kernel_config->datatype_size_out );
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_aarch64_store_C( libxsmm_generated_code*            io_generated_code,
                                                                       const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                       const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                       const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                       const unsigned int                 i_n_blocking,
                                                                       const unsigned int                 i_m_blocking,
                                                                       const unsigned int                 i_use_masking,
                                                                       const unsigned int                 i_packed_width ) {
  unsigned int l_n = 0;
  unsigned int l_m = 0;

  /* store C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    for ( l_m = 0; l_m < i_m_blocking; l_m++ ) {
      libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2,
                                                        (l_n*i_m_blocking) + l_m, i_micro_kernel_config->datatype_size_out, i_use_masking,
                                                        0, 1, LIBXSMM_GENERATOR_PACKED_GEMM_AARCH64_MASK_REG );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                     i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                     (long long)i_packed_width*i_micro_kernel_config->datatype_size_out );
    }
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                   (long long)(i_xgemm_desc->ldc-i_m_blocking)*i_packed_width*i_micro_kernel_config->datatype_size_out  );
  }
  /* reset C pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_c, i_gp_reg_mapping->gp_reg_help_2, i_gp_reg_mapping->gp_reg_c,
                                                 (long long)i_n_blocking*i_xgemm_desc->ldc*i_packed_width*i_micro_kernel_config->datatype_size_out );
}

LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_aarch64_kloop_simd_packed( libxsmm_generated_code*            io_generated_code,
                                                                                 libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                 const unsigned int                 i_packed_width,
                                                                                 const unsigned int                 i_simd_packed_width,
                                                                                 const unsigned int                 i_simd_packed_valid,
                                                                                 const unsigned int                 i_n_blocking,
                                                                                 const unsigned int                 i_m_blocking ) {
  unsigned int l_n = 0;
  unsigned int l_m = 0;
  unsigned int l_use_masking = 0;
  unsigned int l_is_sve = ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT) ) ? 1 : 0;

  /* check if we need to compute a mask */
  if ( i_simd_packed_width > i_simd_packed_valid ) {
    l_use_masking = (l_is_sve) ? 1 : i_simd_packed_valid;
  }

  /* load C accumulator */
  libxsmm_generator_packed_gemm_aarch64_load_C( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_n_blocking, i_m_blocking, l_use_masking, i_packed_width );

  /* k loop */
  libxsmm_generator_loop_header_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, i_xgemm_desc->k );

  /* full vector load of A */
  for ( l_m = 0; l_m < i_m_blocking; ++l_m ) {
    libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0,
                                                      i_n_blocking*i_m_blocking + l_m, i_micro_kernel_config->datatype_size_out, l_use_masking,
                                                      0, 0, LIBXSMM_GENERATOR_PACKED_GEMM_AARCH64_MASK_REG );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                   (long long)i_packed_width*i_micro_kernel_config->datatype_size_in );
  }

  /* loop over the register block */
  for ( l_n = 0; l_n < i_n_blocking; ++l_n ) {
    /* load B */
    libxsmm_generator_vloadstore_masked_vreg_aarch64( io_generated_code, i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1,
                                                      i_n_blocking*i_m_blocking + i_m_blocking, i_micro_kernel_config->datatype_size_out, l_use_masking,
                                                      0, 0, LIBXSMM_GENERATOR_PACKED_GEMM_AARCH64_MASK_REG );
    libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                   i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                   (long long)i_xgemm_desc->ldb * i_packed_width * i_micro_kernel_config->datatype_size_in );
    for ( l_m = 0; l_m < i_m_blocking; ++l_m ) {
      if ( l_is_sve ) {
        libxsmm_aarch64_instruction_sve_compute( io_generated_code,
                                                 LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P,
                                                 i_n_blocking*i_m_blocking + l_m, i_n_blocking*i_m_blocking + i_m_blocking,
                                                 (unsigned char)-1,
                                                 (l_n * i_m_blocking) + l_m, LIBXSMM_AARCH64_SVE_REG_P0,
                                                 (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_SVE_TYPE_S : LIBXSMM_AARCH64_SVE_TYPE_D );
      } else {
        libxsmm_aarch64_instruction_asimd_compute( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,
                                                   i_n_blocking*i_m_blocking + l_m, i_n_blocking*i_m_blocking + i_m_blocking, 0, (l_n * i_m_blocking) + l_m,
                                                   (i_micro_kernel_config->datatype_size_in == 4) ? LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S : LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D );
      }
    }
  }

  /* advance A pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD,
                                                 i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                 (long long)(i_xgemm_desc->lda - i_m_blocking)*i_packed_width*i_micro_kernel_config->datatype_size_in );

  /* advance B pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                 (long long)(   ((long long)i_n_blocking * i_xgemm_desc->ldb * i_packed_width * i_micro_kernel_config->datatype_size_in)
                                                              - ((long long)i_packed_width * i_micro_kernel_config->datatype_size_in)) );

  /* close k loop */
  libxsmm_generator_loop_footer_aarch64( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping->gp_reg_kloop, 1 );

  /* store C accumulator */
  libxsmm_generator_packed_gemm_aarch64_store_C( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, i_n_blocking, i_m_blocking, l_use_masking, i_packed_width );

  /* reset A pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_help_0, i_gp_reg_mapping->gp_reg_a,
                                                 (long long)i_xgemm_desc->k * i_xgemm_desc->lda * i_packed_width * i_micro_kernel_config->datatype_size_in );

  /* reset B Pointer */
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 i_gp_reg_mapping->gp_reg_b, i_gp_reg_mapping->gp_reg_help_1, i_gp_reg_mapping->gp_reg_b,
                                                 (long long)i_xgemm_desc->k * i_packed_width * i_micro_kernel_config->datatype_size_in );
}

#ifdef LIBXSMM_GENERATOR_PACKED_GEMM_AARCH64_MASK_REG
#undef LIBXSMM_GENERATOR_PACKED_GEMM_AARCH64_MASK_REG
#endif
