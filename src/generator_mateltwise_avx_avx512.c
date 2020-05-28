/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_mateltwise_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_header_m_loop( libxsmm_generated_code*                io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*   i_kernel_config,
                                              const unsigned int                        i_gp_reg_m_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_add_instruction, i_gp_reg_m_loop, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_m_loop( libxsmm_generated_code*                    io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*       i_kernel_config,
                                              const unsigned int                            i_gp_reg_m_loop,
                                              const unsigned int                            i_m ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_cmp_instruction, i_gp_reg_m_loop, i_m );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_header_n_loop( libxsmm_generated_code*                io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*   i_kernel_config,
                                              const unsigned int                        i_gp_reg_n_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_add_instruction, i_gp_reg_n_loop, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_n_loop( libxsmm_generated_code*                    io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*       i_kernel_config,
                                              const unsigned int                            i_gp_reg_n_loop,
                                              const unsigned int                            i_n ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_cmp_instruction, i_gp_reg_n_loop, i_n );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_initialize_avx512_mask( libxsmm_generated_code*            io_generated_code,
    const unsigned int                       i_gp_reg_tmp,
    const libxsmm_meltw_descriptor*          i_mateltwise_desc,
    const unsigned int                       i_mask_reg,
    const unsigned int                       i_mask_count,
    const unsigned int                       i_precision) {

  unsigned int l_mask;
  if (i_precision == LIBXSMM_GEMM_PRECISION_F32) {
    l_mask = 0xffff;
  } else if (i_precision == LIBXSMM_GEMM_PRECISION_BF16) {
    l_mask = 0xffffffff;
  } else if (i_precision == LIBXSMM_GEMM_PRECISION_I8) {
    l_mask = 0xffffffffffffffff;
  } 
  /* shift right by "inverse" remainder */
  l_mask = l_mask >> i_mask_count;

  /* move mask to GP register */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
      LIBXSMM_X86_INSTR_MOVQ,
      i_gp_reg_tmp,
      l_mask );
  
  if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE  ) {
    if ( LIBXSMM_GEMM_PRECISION_F32 == i_precision ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVW,
          i_gp_reg_tmp,
          i_mask_reg, 0 );
    } else if ( LIBXSMM_GEMM_PRECISION_BF16 == i_precision ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVD,
          i_gp_reg_tmp,
          i_mask_reg, 0 );
    } else if ( LIBXSMM_GEMM_PRECISION_I8 == i_precision ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVQ,
          i_gp_reg_tmp,
          i_mask_reg, 0 );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    /* shouldn't happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( libxsmm_mateltwise_kernel_config*    io_micro_kernel_config,
    const unsigned int              i_arch,
    const libxsmm_meltw_descriptor* i_mateltwise_desc) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  if ( i_arch >= LIBXSMM_X86_AVX512_CORE ) { 
    io_micro_kernel_config->instruction_set = LIBXSMM_X86_AVX512_CORE;
    io_micro_kernel_config->vector_reg_count = 16;
    /* Configure input specific microkernel options */
    if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vector_length_in = 16;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vector_length_in = 32;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->vector_length_in = 64;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU8;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    /* Configure output specific microkernel options */
    if ( LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vector_length_out = 16;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vector_length_out = 32;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_GEMM_PRECISION_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 1;
      io_micro_kernel_config->vector_length_out = 64;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU8;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
    io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
    io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
    io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
    io_micro_kernel_config->vector_name = 'z';
  } else {
    /* That should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_cvtfp32bf16_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int m, n, m_trips, use_m_masking, mask_in_count, mask_out_count, reg_0, reg_1;

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R11;

  /* load the input pointer and output pointer */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      8,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  /* We fully unroll in M dimension, calculate mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % 32 == 0 ) ? 0 : 1;
  m_trips           = (m + 31) / 32;

  /* Calculate input and output masks in case we see m_masking */
  if (use_m_masking == 1) {
    /* If the remaining elements are < 16, then we read a full vector and a partial one at the last m trip */
    /* If the remaining elements are >= 16, then we read a partial vector at the last m trip  */
    /* Calculate mask reg 1 for input-reading */
    mask_in_count = ( (m % 32) > 16) ? 32 - (m % 32) : 16 - (m % 32);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R12, i_mateltwise_desc, 1, mask_in_count, LIBXSMM_GEMM_PRECISION_F32);
    /* Calculate mask reg 2 for output-writing */
    mask_out_count = 32 - (m % 32);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R12, i_mateltwise_desc, 2, mask_out_count, LIBXSMM_GEMM_PRECISION_BF16);
  }

  if (n > 1) {
    /* open n loop */
    libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
  }

  for (im = 0; im < m_trips; im++) {
    /* Load in zmm_i and zmm_i+16 the two input fp32 zmms  */
    reg_0 = im % 15;
    reg_1 = (im % 15)+16;
    
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        i_micro_kernel_config->vmove_instruction_in,
        i_gp_reg_mapping->gp_reg_in,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        im * 32 * i_micro_kernel_config->datatype_size_in,
        i_micro_kernel_config->vector_name,
        reg_0, ((im == (m_trips-1)) && (m % 32 < 16)) ? use_m_masking : 0, 1, 0 );

    /* If last iteration and remainder is less than 16, do not load anything  */
   if (!((use_m_masking == 1) && (im == m_trips-1) && (m % 32 <= 16))) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * 32 + 16) * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          reg_1, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
   }

    /* Downconvert to BF16  */
    libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
        i_micro_kernel_config->instruction_set,
        LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
        i_micro_kernel_config->vector_name,
        reg_0, reg_1,
        reg_0,
        0);

    /* Store the result  */
    libxsmm_x86_instruction_vec_move( io_generated_code,
        i_micro_kernel_config->instruction_set,
        i_micro_kernel_config->vmove_instruction_out,
        i_gp_reg_mapping->gp_reg_out,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        im * 32 * i_micro_kernel_config->datatype_size_out,
        i_micro_kernel_config->vector_name,
        reg_0, (im == (m_trips-1)) ? use_m_masking : 0, 2, 1 );  
  }

  if (n > 1) {
    /* Adjust input and output pointer */
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_in,
        i_mateltwise_desc->ldi *  i_micro_kernel_config->datatype_size_in);

    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_out,
        i_mateltwise_desc->ldo *  i_micro_kernel_config->datatype_size_out);
    
    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_avx_avx512_kernel( libxsmm_generated_code*         io_generated_code,
    const libxsmm_meltw_descriptor* i_mateltwise_desc) {
  libxsmm_mateltwise_kernel_config  l_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker        l_loop_label_tracker;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping)); 
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif

  /* define mateltwise kernel config */
  libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( &l_kernel_config, io_generated_code->arch, i_mateltwise_desc);

  /* open asm */
  libxsmm_x86_instruction_open_stream_mateltwise( io_generated_code, l_gp_reg_mapping.gp_reg_param_struct, io_generated_code->arch );

  /* Depending on the elementwise function, dispatch the proper code JITer */
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16) {
    libxsmm_generator_cvtfp32bf16_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_mateltwise( io_generated_code, io_generated_code->arch );
}

