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
#include "generator_packed_gemm_ac_rm_avx_avx2_avx512.h"
#include "generator_gemm_common.h"
#include "generator_x86_instructions.h"
#include "libxsmm_main.h"

LIBXSMM_API void libxsmm_generator_packed_gemm_ac_rm( libxsmm_generated_code*         io_generated_code,
                                                      const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                      const unsigned int              i_packed_width,
                                                      const char*                     i_arch ) {
  if ( strcmp(i_arch, "knl") == 0 ||
       strcmp(i_arch, "knm") == 0 ||
       strcmp(i_arch, "skx") == 0 ||
       strcmp(i_arch, "clx") == 0 ||
       strcmp(i_arch, "cpx") == 0 ||
       strcmp(i_arch, "hsw") == 0 ||
       strcmp(i_arch, "snb") == 0 ) {
    if ( strcmp(i_arch, "snb") == 0 ) {
      io_generated_code->arch = LIBXSMM_X86_AVX;
    } else if ( strcmp(i_arch, "hsw") == 0 ) {
      io_generated_code->arch = LIBXSMM_X86_AVX2;
    } else if ( strcmp(i_arch, "knl") == 0 ) {
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

    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512( io_generated_code,
                                                         i_xgemm_desc, i_packed_width );
  } else {
    fprintf( stderr, "RM AC SOA is only available for AVX/AVX2/AVX512 at this point\n" );
    exit(-1);
  }
}

LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512( libxsmm_generated_code*         io_generated_code,
                                                                             const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                             const unsigned int              i_packed_width ) {
  unsigned int l_max_reg_block = 0;
  unsigned int l_n1_range = 0;
  unsigned int l_n2_range = 0;
  unsigned int l_n1_block = 0;
  unsigned int l_n2_block = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  /* select accumulator blocking */
  /* @TODO we could do more agressive blockings if needed */
  if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
    l_max_reg_block = 28;
  } else {
    l_max_reg_block = 13;
  }

  /* define gp register mapping */
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
/*  l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_X86_GP_REG_R9;*/
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R10;     /* this is the SIMD packed register loop */
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_R11;     /* this is the gp register to load the AVX512 mask, if needed */
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, io_generated_code->arch, i_xgemm_desc, 0 );

  /* calculate the chunk size of current columns to work on */
  if ( libxsmm_compute_equalized_blocking( i_xgemm_desc->n, l_max_reg_block, &l_n1_range, &l_n1_block, &l_n2_range, &l_n2_block ) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );

  /* m loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_mloop, 0 );
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_mloop, 1 );

  /* loop over n-blocks */
  if ( l_n1_block == i_xgemm_desc->n ) {
    /* no N loop at all */
    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                               i_packed_width, i_xgemm_desc->n );
  } else if ( (l_n1_range > 0) && (l_n2_range > 0) ) {
    /* reset n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_nloop, 0 );

    /* we have two ranges */
    /* first range */
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n1_block );

    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                               i_packed_width, l_n1_block );

    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n1_range );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

    /* second range */
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n2_block );

    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                               i_packed_width, l_n2_block );

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
                                     i_xgemm_desc->n * i_packed_width * l_micro_kernel_config.datatype_size );
  } else if ( (l_n1_range > 0) && (l_n2_range == 0) ) {
    /* reset n loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_nloop, 0 );

    /* we have one range */
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, &l_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_nloop, l_n1_block );

    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_micro_kernel_config, i_xgemm_desc,
                                                               i_packed_width, l_n1_block );

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
                                     i_xgemm_desc->n * i_packed_width * l_micro_kernel_config.datatype_size );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_N_BLOCK );
    return;
  }

  /* advance A pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_a,
                                   l_micro_kernel_config.datatype_size*i_packed_width*i_xgemm_desc->lda);

  /* advance C pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_c,
                                   l_micro_kernel_config.datatype_size*i_packed_width*i_xgemm_desc->ldc);

  /* close m loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_xgemm_desc->prefetch );
}


LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop( libxsmm_generated_code*            io_generated_code,
                                                                                   libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                                   const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                   const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                   const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                   const unsigned int                 i_packed_width,
                                                                                   const unsigned int                 i_n_blocking ) {
  /* calculate how many iterations we need */
  unsigned int l_simd_packed_remainder = 0;
  unsigned int l_simd_packed_iters = 0;
  unsigned int l_simd_packed_width = 0;

  /* select simd packing width and accumulator blocking */
  if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
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
  l_simd_packed_remainder = i_packed_width % l_simd_packed_width;
  l_simd_packed_iters = i_packed_width/l_simd_packed_width;

  /* check if we have a single SIMD devisor */
  if ( l_simd_packed_width == i_packed_width ) {
    /* run inner compute kernel */
    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop_simd_packed( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                                           i_packed_width, l_simd_packed_width, l_simd_packed_width, i_n_blocking );
  /* check if we have a perfect SIMD devisor */
  } else if ( l_simd_packed_remainder == 0 ) {
    /* initilize packed loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, l_simd_packed_width );

    /* run inner compute kernel */
    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop_simd_packed( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                                           i_packed_width, l_simd_packed_width, l_simd_packed_width, i_n_blocking );

    /* advance A pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     l_simd_packed_width * i_micro_kernel_config->datatype_size );

    /* advance C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_c,
                                     l_simd_packed_width * i_micro_kernel_config->datatype_size );

    /* jump back to pack loop label */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_help_0, i_packed_width );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

    /* reset A pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_sub_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     l_simd_packed_iters * l_simd_packed_width * i_micro_kernel_config->datatype_size );

    /* reset C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_sub_instruction,
                                     i_gp_reg_mapping->gp_reg_c,
                                     l_simd_packed_iters * l_simd_packed_width * i_micro_kernel_config->datatype_size );
  /* we need masking and have less than SIMD width */
  } else if ( l_simd_packed_width > i_packed_width  ) {
    /* run remainder compute kernel */
    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop_simd_packed( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                                           i_packed_width, l_simd_packed_width, l_simd_packed_remainder, i_n_blocking );
  /* we need the general case */
  } else {
    /* initilize packed loop */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_help_0, 0 );
    libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_help_0, l_simd_packed_width );

    /* run inner compute kernel */
    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop_simd_packed( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                                           i_packed_width, l_simd_packed_width, l_simd_packed_width, i_n_blocking );

    /* advance A pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     l_simd_packed_width * i_micro_kernel_config->datatype_size );

    /* advance C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_c,
                                     l_simd_packed_width * i_micro_kernel_config->datatype_size );

    /* jump back to pack loop label */
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_help_0, i_packed_width - l_simd_packed_remainder );
    libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, io_loop_label_tracker );

    /* run remainder compute kernel */
    libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop_simd_packed( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc,
                                                                           i_packed_width, l_simd_packed_width, l_simd_packed_remainder, i_n_blocking );

    /* reset A pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_sub_instruction,
                                     i_gp_reg_mapping->gp_reg_a,
                                     l_simd_packed_iters * l_simd_packed_width * i_micro_kernel_config->datatype_size );

    /* reset C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_sub_instruction,
                                     i_gp_reg_mapping->gp_reg_c,
                                     l_simd_packed_iters * l_simd_packed_width * i_micro_kernel_config->datatype_size );
  }

  /* advance B and C pointers if N is bigger than our register blocking */
  if ( i_xgemm_desc->n != i_n_blocking ) {
    /* advance B pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_b,
                                     i_n_blocking * i_micro_kernel_config->datatype_size );

    /* advance C pointer */
    libxsmm_x86_instruction_alu_imm( io_generated_code,
                                     i_micro_kernel_config->alu_add_instruction,
                                     i_gp_reg_mapping->gp_reg_c,
                                     i_n_blocking * i_packed_width * i_micro_kernel_config->datatype_size );
  }
}


LIBXSMM_API_INTERN void libxsmm_generator_packed_gemm_ac_rm_avx_avx2_avx512_kloop_simd_packed( libxsmm_generated_code*            io_generated_code,
                                                                                               libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                                               const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                               const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                               const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                               const unsigned int                 i_packed_width,
                                                                                               const unsigned int                 i_simd_packed_width,
                                                                                               const unsigned int                 i_simd_packed_valid,
                                                                                               const unsigned int                 i_n_blocking ) {
  unsigned int l_n = 0;
  unsigned int l_lcl_k = 0;
  unsigned int l_use_masking = 0;
  unsigned int l_avx_ac_move_instr = 0;

  /* check if we need to compute a mask */
  if ( i_simd_packed_width > i_simd_packed_valid ) {
    /* on AVX512 we can use mask registers */
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      libxsmm_generator_gemm_initialize_avx512_mask( io_generated_code, i_gp_reg_mapping->gp_reg_help_1, i_xgemm_desc,  i_simd_packed_width - i_simd_packed_valid );
    } else {
      char l_id = (char)i_n_blocking;
      unsigned char l_data[32];
      unsigned int l_count;

      if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
        unsigned long long* l_i64_ptr = (unsigned long long*)l_data;
        for ( l_count = 0; l_count < 4; ++l_count ) {
          if ( l_count < i_simd_packed_valid ) {
            l_i64_ptr[l_count] = 0xffffffffffffffff;
          } else {
            l_i64_ptr[l_count] = 0x0;
          }
        }
        l_avx_ac_move_instr = LIBXSMM_X86_INSTR_VMASKMOVPD;
      } else {
        unsigned int* l_i32_ptr = (unsigned int*)l_data;
        for ( l_count = 0; l_count < 8; ++l_count ) {
          if ( l_count < i_simd_packed_valid ) {
            l_i32_ptr[l_count] = 0xffffffff;
          } else {
            l_i32_ptr[l_count] = 0x0;
          }
        }
        l_avx_ac_move_instr = LIBXSMM_X86_INSTR_VMASKMOVPS;
      }

      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, l_data, &l_id, 'y', 15 );
    }
    l_use_masking = LIBXSMM_X86_AVX512_MASK;
  }

  /* load C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & i_xgemm_desc->flags)) { /* Beta=0 */
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               i_micro_kernel_config->vxor_instruction,
                                               i_micro_kernel_config->vector_name,
                                               l_n, l_n, l_n );
    } else {
      /* in case of masking we need to distinguish between AVX/AVX2 and AVX512 */
      if ( l_use_masking ) {
        if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->c_vmove_instruction,
                                            i_gp_reg_mapping->gp_reg_c,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            l_n*i_packed_width*i_micro_kernel_config->datatype_size,
                                            i_micro_kernel_config->vector_name,
                                            l_n, l_use_masking, 1, 0 );
        } else {
          libxsmm_x86_instruction_vec_mask_move( io_generated_code,
                                                 l_avx_ac_move_instr,
                                                 i_gp_reg_mapping->gp_reg_c,
                                                 LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                 l_n*i_packed_width*i_micro_kernel_config->datatype_size,
                                                 i_micro_kernel_config->vector_name,
                                                 l_n, 15, 0);
        }
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->c_vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_n*i_packed_width*i_micro_kernel_config->datatype_size,
                                          i_micro_kernel_config->vector_name,
                                          l_n, 0, 1, 0 );
      }
    }
  }

  /* k loop */
  libxsmm_generator_gemm_header_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, 0, 1 );

  /* full vector load of A */
  /* @TODO: prepare KNM's QMADD */
  for ( l_lcl_k = 0; l_lcl_k < 1; l_lcl_k++ ) {
    /* in case of masking we need to distinguish between AVX/AVX2 and AVX512 */
    if ( l_use_masking ) {
      if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->a_vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_lcl_k*i_packed_width*i_micro_kernel_config->datatype_size,
                                          i_micro_kernel_config->vector_name,
                                          i_n_blocking+l_lcl_k, l_use_masking, 1, 0 );
      } else {
        libxsmm_x86_instruction_vec_mask_move( io_generated_code,
                                               l_avx_ac_move_instr,
                                               i_gp_reg_mapping->gp_reg_a,
                                               LIBXSMM_X86_GP_REG_UNDEF, 0,
                                               l_lcl_k*i_packed_width*i_micro_kernel_config->datatype_size,
                                               i_micro_kernel_config->vector_name,
                                               i_n_blocking+l_lcl_k, 15, 0);
      }
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->a_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_a,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_lcl_k*i_packed_width*i_micro_kernel_config->datatype_size,
                                        i_micro_kernel_config->vector_name,
                                        i_n_blocking+l_lcl_k, 0, 1, 0 );
    }
  }

  /* loop over the register block */
  for ( l_n = 0; l_n < i_n_blocking; ++l_n ) {
    if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
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
    } else if ( io_generated_code->arch == LIBXSMM_X86_AVX2 ) {
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
    } else if ( io_generated_code->arch == LIBXSMM_X86_AVX ) {
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
    } else {
      /* cannot happen */
    }
  }

  /* advance A pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_a,
                                   i_packed_width * i_micro_kernel_config->datatype_size );

  /* advance B pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_add_instruction,
                                   i_gp_reg_mapping->gp_reg_b,
                                   i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size );

  /* close k loop */
  libxsmm_generator_gemm_footer_kloop( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, 0, i_xgemm_desc->k, 0 );

  /* store C accumulator */
  for ( l_n = 0; l_n < i_n_blocking; l_n++ ) {
    /* in case of masking we need to distinguish between AVX/AVX2 and AVX512 */
    if ( l_use_masking ) {
      if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->c_vmove_instruction,
                                          i_gp_reg_mapping->gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_n*i_packed_width*i_micro_kernel_config->datatype_size,
                                          i_micro_kernel_config->vector_name,
                                          l_n, l_use_masking, 0, 1 );
      } else {
        libxsmm_x86_instruction_vec_mask_move( io_generated_code,
                                               l_avx_ac_move_instr,
                                               i_gp_reg_mapping->gp_reg_c,
                                               LIBXSMM_X86_GP_REG_UNDEF, 0,
                                               l_n*i_packed_width*i_micro_kernel_config->datatype_size,
                                               i_micro_kernel_config->vector_name,
                                               l_n, 15, 1);
      }
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->c_vmove_instruction,
                                        i_gp_reg_mapping->gp_reg_c,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        l_n*i_packed_width*i_micro_kernel_config->datatype_size,
                                        i_micro_kernel_config->vector_name,
                                        l_n, 0, 0, 1 );
    }
  }

  /* reset A pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_sub_instruction,
                                   i_gp_reg_mapping->gp_reg_a,
                                   i_xgemm_desc->k * i_packed_width * i_micro_kernel_config->datatype_size );

  /* reset B Pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
                                   i_micro_kernel_config->alu_sub_instruction,
                                   i_gp_reg_mapping->gp_reg_b,
                                   i_xgemm_desc->k * i_xgemm_desc->ldb * i_micro_kernel_config->datatype_size );
}
