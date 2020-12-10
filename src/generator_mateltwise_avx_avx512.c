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
#include "generator_mateltwise_transform_avx_avx512.h"
#include "generator_mateltwise_dropout_avx_avx512.h"
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
void libxsmm_generator_mateltwise_header_n_dyn_loop( libxsmm_generated_code*                io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*   i_kernel_config,
                                              const unsigned int                        i_gp_reg_n_loop,
                                              int                                       skip_init ) {
  if (skip_init == 0) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  }
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_n_dyn_loop( libxsmm_generated_code*                    io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*       i_kernel_config,
                                              const unsigned int                            i_gp_reg_n_loop,
                                              const unsigned int                            i_gp_reg_n_bound ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_add_instruction, i_gp_reg_n_loop, 1);
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_kernel_config->alu_cmp_instruction, i_gp_reg_n_bound, i_gp_reg_n_loop);
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_haddps_avx512( libxsmm_generated_code*                        io_generated_code,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const unsigned int                             i_vec_inout,
    const unsigned int                             i_vec_tmp1,
    const unsigned int                             i_vec_tmp2) {

  if (i_vec_tmp1 > 15 || i_vec_tmp2 > 15 ) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           i_vec_inout, i_vec_inout, i_vec_tmp1, 0x4e );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           i_vec_inout, i_vec_tmp1, i_vec_inout );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           i_micro_kernel_config->vector_name,
                                           i_vec_inout, i_vec_inout, i_vec_tmp1, 0xb1 );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           i_vec_inout, i_vec_tmp1, i_vec_tmp2 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp2, i_vec_tmp1, 0x4e );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp1, i_vec_tmp2 );

  libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp2, i_vec_tmp1, 0x1 );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           i_micro_kernel_config->vector_name,
                                           i_vec_tmp2, i_vec_tmp1, i_vec_inout );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_initialize_avx512_mask( libxsmm_generated_code*            io_generated_code,
    const unsigned int                       i_gp_reg_tmp,
    const unsigned int                       i_mask_reg,
    const unsigned int                       i_mask_count,
    const unsigned int                       i_precision) {

  unsigned long long l_mask = 0;

  if ( i_precision == LIBXSMM_DATATYPE_F64 || i_precision == LIBXSMM_DATATYPE_I64 ) {
    l_mask = 0xff;
  } else if ( i_precision == LIBXSMM_DATATYPE_F32 || i_precision == LIBXSMM_DATATYPE_I32 ) {
    l_mask = 0xffff;
  } else if ( i_precision == LIBXSMM_DATATYPE_F16 || i_precision == LIBXSMM_DATATYPE_BF16 || i_precision == LIBXSMM_DATATYPE_I16 ) {
    l_mask = 0xffffffff;
  } else if ( i_precision == LIBXSMM_GEMM_PRECISION_I8 ) {
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
    if ( i_precision == LIBXSMM_DATATYPE_F64 || i_precision == LIBXSMM_DATATYPE_I64 ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
          i_gp_reg_tmp,
          i_mask_reg );
    } else if ( i_precision == LIBXSMM_DATATYPE_F32 || i_precision == LIBXSMM_DATATYPE_I32 ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVW_GPR_LD,
          i_gp_reg_tmp,
          i_mask_reg );
    } else if ( i_precision == LIBXSMM_DATATYPE_F16 || i_precision == LIBXSMM_DATATYPE_BF16 || i_precision == LIBXSMM_DATATYPE_I16 ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
          i_gp_reg_tmp,
          i_mask_reg );
    } else if ( i_precision == LIBXSMM_GEMM_PRECISION_I8 ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVQ_GPR_LD,
          i_gp_reg_tmp,
          i_mask_reg );
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
void libxsmm_generator_mateltwise_xoshiro128pp_avx512( libxsmm_generated_code* io_generated_code,
                                                       const unsigned int      i_vec_reg_rng_state_0,
                                                       const unsigned int      i_vec_reg_rng_state_1,
                                                       const unsigned int      i_vec_reg_rng_state_2,
                                                       const unsigned int      i_vec_reg_rng_state_3,
                                                       const unsigned int      i_vec_reg_rng_tmp_0,
                                                       const unsigned int      i_vec_reg_rng_tmp_1,
                                                       const unsigned int      o_vec_reg_rng ) {
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPADDD, 'z',
                                            i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, i_vec_reg_rng_tmp_0 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z',
                                                 i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, 7 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, 'z',
                                                 i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_0, 25 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, 'z',
                                            i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, i_vec_reg_rng_tmp_0 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPADDD, 'z',
                                            i_vec_reg_rng_tmp_0, i_vec_reg_rng_state_0, o_vec_reg_rng);

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z',
                                                 i_vec_reg_rng_state_1, i_vec_reg_rng_tmp_0, 9);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                            i_vec_reg_rng_state_2, i_vec_reg_rng_state_0, i_vec_reg_rng_state_2 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                            i_vec_reg_rng_state_3, i_vec_reg_rng_state_1, i_vec_reg_rng_state_3 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                            i_vec_reg_rng_state_1, i_vec_reg_rng_state_2, i_vec_reg_rng_state_1 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                            i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, i_vec_reg_rng_state_0 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                            i_vec_reg_rng_state_2, i_vec_reg_rng_tmp_0, i_vec_reg_rng_state_2 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z',
                                                 i_vec_reg_rng_state_3, i_vec_reg_rng_tmp_0, 11 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, 'z',
                                                 i_vec_reg_rng_state_3, i_vec_reg_rng_tmp_1, 21 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, 'z',
                                            i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, i_vec_reg_rng_state_3);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_xoshiro128p_f32_avx512( libxsmm_generated_code* io_generated_code,
                                                          const unsigned int      i_vec_reg_rng_state_0,
                                                          const unsigned int      i_vec_reg_rng_state_1,
                                                          const unsigned int      i_vec_reg_rng_state_2,
                                                          const unsigned int      i_vec_reg_rng_state_3,
                                                          const unsigned int      i_vec_reg_rng_tmp_0,
                                                          const unsigned int      i_vec_reg_rng_tmp_1,
                                                          const unsigned int      i_vec_reg_rng_one,
                                                          const unsigned int      o_vec_reg_rng ) {
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPADDD, 'z',
                                            i_vec_reg_rng_state_3, i_vec_reg_rng_state_0, o_vec_reg_rng);

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, 'z',
                                                 o_vec_reg_rng, o_vec_reg_rng, 9);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, 'z',
                                            o_vec_reg_rng, i_vec_reg_rng_one, o_vec_reg_rng);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, 'z',
                                            i_vec_reg_rng_one, o_vec_reg_rng, o_vec_reg_rng);

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z',
                                                 i_vec_reg_rng_state_1, i_vec_reg_rng_tmp_0, 9);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                            i_vec_reg_rng_state_2, i_vec_reg_rng_state_0, i_vec_reg_rng_state_2 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                            i_vec_reg_rng_state_3, i_vec_reg_rng_state_1, i_vec_reg_rng_state_3 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                            i_vec_reg_rng_state_1, i_vec_reg_rng_state_2, i_vec_reg_rng_state_1 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                            i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, i_vec_reg_rng_state_0 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, 'z',
                                            i_vec_reg_rng_state_2, i_vec_reg_rng_tmp_0, i_vec_reg_rng_state_2 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z',
                                                 i_vec_reg_rng_state_3, i_vec_reg_rng_tmp_0, 11 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, 'z',
                                                 i_vec_reg_rng_state_3, i_vec_reg_rng_tmp_1, 21 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, 'z',
                                            i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, i_vec_reg_rng_state_3);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( libxsmm_generated_code*         io_generated_code,
    libxsmm_mateltwise_kernel_config*    io_micro_kernel_config,
    const unsigned int              i_arch,
    const libxsmm_meltw_descriptor* i_mateltwise_desc) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  if ( i_arch >= LIBXSMM_X86_AVX512_CORE ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_X86_AVX512_CORE;
    io_micro_kernel_config->vector_reg_count = 16;
    /* Configure input specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->vector_length_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vector_length_in = 16;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vector_length_in = 32;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vector_length_in = 16;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->vector_length_in = 64;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU8;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    /* Configure output specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->vector_length_out = 8;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVUPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vector_length_out = 16;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vector_length_out = 32;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vector_length_out = 16;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
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
void libxsmm_generator_tanh_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_x2,
    const unsigned int                             i_vec_nom,
    const unsigned int                             i_vec_denom,
    const unsigned int                             i_mask_hi,
    const unsigned int                             i_mask_lo,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_c1_d,
    const unsigned int                             i_vec_c2_d,
    const unsigned int                             i_vec_c3_d,
    const unsigned int                             i_vec_hi_bound,
    const unsigned int                             i_vec_lo_bound,
    const unsigned int                             i_vec_ones,
    const unsigned int                             i_vec_neg_ones
    ) {
  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        i_micro_kernel_config->vector_name,
                                        i_vec_x, i_vec_x, i_vec_x2 );

  libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VCMPPS,
                                                i_micro_kernel_config->vector_name,
                                                i_vec_hi_bound,
                                                i_vec_x,
                                                i_vec_x,
                                                17, i_mask_hi, 0 );

  libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VCMPPS,
                                                i_micro_kernel_config->vector_name,
                                                i_vec_lo_bound,
                                                i_vec_x,
                                                i_vec_x,
                                                30, i_mask_lo, 0 );

   libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VMOVDQU64,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_x2, LIBXSMM_X86_VEC_REG_UNDEF, i_vec_nom );

   libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c2, i_vec_c3, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c1, i_vec_x2, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c0, i_vec_x2, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        i_micro_kernel_config->vector_name,
                                        i_vec_nom, i_vec_x, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VADDPS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_x2, i_vec_c3_d, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c2_d, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c1_d, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_c0, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                       i_micro_kernel_config->instruction_set,
                                       LIBXSMM_X86_INSTR_VRCP14PS,
                                       i_micro_kernel_config->vector_name,
                                       i_vec_denom, LIBXSMM_X86_VEC_REG_UNDEF, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        i_micro_kernel_config->vector_name,
                                        i_vec_denom, i_vec_nom, i_vec_x );

  libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VBLENDMPS,
                                                i_micro_kernel_config->vector_name,
                                                i_vec_x,
                                                i_vec_ones,
                                                i_vec_x,
                                                LIBXSMM_X86_IMM_UNDEF,
                                                i_mask_hi, 0 );

  libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VBLENDMPS,
                                                i_micro_kernel_config->vector_name,
                                                i_vec_x,
                                                i_vec_neg_ones,
                                                i_vec_x,
                                                LIBXSMM_X86_IMM_UNDEF,
                                                i_mask_lo, 0 );
}


LIBXSMM_API_INTERN
void libxsmm_generator_cvtfp32bf16_avx512_replacement_sequence( libxsmm_generated_code*                        io_generated_code,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const unsigned int                             i_vec_reg,
    const unsigned int                             tmp1,
    const unsigned int                             tmp2) {

  /* and with naninf */
  libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_VPANDD,
      1,
      LIBXSMM_X86_GP_REG_RSP,
      LIBXSMM_X86_GP_REG_UNDEF,
      0,
      24,
      i_micro_kernel_config->vector_name,
      i_vec_reg,
      tmp1 );

  /* and with fixup */
  libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_VPANDD,
      1,
      LIBXSMM_X86_GP_REG_RSP,
      LIBXSMM_X86_GP_REG_UNDEF,
      0,
      16,
      i_micro_kernel_config->vector_name,
      i_vec_reg,
      tmp2 );

  /* compute naninf mask k7 */
  libxsmm_x86_instruction_vec_compute_mem_mask( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_VPCMPD,
      1,
      LIBXSMM_X86_GP_REG_RSP,
      LIBXSMM_X86_GP_REG_UNDEF,
      0,
      24,
      i_micro_kernel_config->vector_name,
      tmp1,
      LIBXSMM_X86_VEC_REG_UNDEF,
      4,
      7, 0 );

  /* compute fixup mask k6 */
  libxsmm_x86_instruction_vec_compute_mem_mask( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_VPCMPD,
      1,
      LIBXSMM_X86_GP_REG_RSP,
      LIBXSMM_X86_GP_REG_UNDEF,
      0,
      16,
      i_micro_kernel_config->vector_name,
      tmp2,
      LIBXSMM_X86_VEC_REG_UNDEF,
      0,
      6, 0 );

  /* load rneadd */
  libxsmm_x86_instruction_vec_move( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_VBROADCASTSS,
      LIBXSMM_X86_GP_REG_RSP,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      8,
      i_micro_kernel_config->vector_name,
      tmp1, 0, 1, 0 );

  /* load fixup */
  libxsmm_x86_instruction_vec_move( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_VBROADCASTSS,
      LIBXSMM_X86_GP_REG_RSP,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_micro_kernel_config->vector_name,
      tmp2, 0, 1, 0 );

  /* compute fixup */
  libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_VPADDD,
      i_micro_kernel_config->vector_name,
      tmp2,
      tmp1,
      tmp1,
      LIBXSMM_X86_IMM_UNDEF,
      6,
      0 );

  /* compute fixup */
  libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
      i_micro_kernel_config->instruction_set,
      LIBXSMM_X86_INSTR_VPADDD,
      i_micro_kernel_config->vector_name,
      tmp1,
      i_vec_reg,
      i_vec_reg,
      LIBXSMM_X86_IMM_UNDEF,
      7,
      0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_cvtfp32bf16_vnni_format_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int m, n, im, in, unroll_iter, use_m_masking = 0, m_trips = 0, n_trips = 0, vlen_in = 16, vlen_out = 32, mask_in_count = 0;
  unsigned int reserved_zmms = 1, max_nm_unrolling = 31, n_unroll_factor = 2;
  unsigned int mask_reg_in = 1, mask_reg_out = 2;
  unsigned int perm_vreg = 0, vreg_in_0 = 0, vreg_in_1 = 0;
  short vnni_perm_array[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};

  /* Some rudimentary checking of M, N and LDs*/
  if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) || (i_mateltwise_desc->m > i_mateltwise_desc->ldo) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* n % 2 should be zero in order to support vnni reformating  */
  if ( i_mateltwise_desc->n % 2 != 0  ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  vlen_in = 16;
  vlen_out = 32;

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R10;

  /* Load permute register */
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) vnni_perm_array, "vnni_perm_array_", i_micro_kernel_config->vector_name, perm_vreg);

  /* Load the input pointer and output pointer */
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
  use_m_masking     = ( m % vlen_in == 0 ) ? 0 : 1;
  m_trips           = (m + vlen_in - 1) / vlen_in;

  max_nm_unrolling = max_nm_unrolling - reserved_zmms;

  if (m_trips > max_nm_unrolling) {
    n_unroll_factor = 2;
  } else {
    /* Explore n unrolling opportunities... We unroll only by factors that divide N  */
    n_unroll_factor = n;
    while (m_trips * n_unroll_factor > max_nm_unrolling) {
      n_unroll_factor = n_unroll_factor - 2;
    }
    while (n % n_unroll_factor > 0) {
      n_unroll_factor = n_unroll_factor - 2;
    }
  }
  n_trips = n / n_unroll_factor;

  /* Calculate input and output masks in case we see m_masking */
  if (use_m_masking == 1) {
    mask_in_count = vlen_in - m % vlen_in;
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R11, mask_reg_in, mask_in_count, LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype));
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R11, mask_reg_out, 2*mask_in_count, LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype));
  }

  if (n_trips > 1) {
    /* open n loop */
    libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
  }

  for (in = 0; in < n_unroll_factor; in+=2) {
    for (im = 0; im < m_trips; im++) {
      unroll_iter = in * m_trips + im * 2;
      vreg_in_0 = unroll_iter % (max_nm_unrolling - reserved_zmms) + reserved_zmms;
      vreg_in_1 = vreg_in_0 + 1;

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * vlen_in + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          vreg_in_1, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_in : 0, 1, 0 );

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * vlen_in + (in+1) * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          vreg_in_0, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_in : 0, 1, 0 );

      libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
          i_micro_kernel_config->vector_name,
          vreg_in_1, vreg_in_0,
          vreg_in_0,
          0);

      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPERMW,
                                               i_micro_kernel_config->vector_name,
                                               vreg_in_0,
                                               perm_vreg,
                                               vreg_in_0);


      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_out,
          i_gp_reg_mapping->gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * vlen_out + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
          i_micro_kernel_config->vector_name,
          vreg_in_0, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_out : 0, 0, 1 );

    }
  }

  if (n_trips > 1) {
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_in,
        i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);

    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_out,
        i_mateltwise_desc->ldo * n_unroll_factor * i_micro_kernel_config->datatype_size_out);

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
  }

}

LIBXSMM_API_INTERN
void libxsmm_generator_cvtfp32bf16_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int i = 0, im, in, m, n, n_trips, m_trips, use_m_masking, mask_in_count, mask_out_count, reg_0, reg_1, unroll_iter = 0, zero_reg = 0;
  unsigned int reserved_zmms = 0, max_nm_unrolling = 31, reserved_mask_regs = 1, current_mask_reg = 1;
  unsigned int n_unroll_factor = 1, eager_result_store = 0;
  unsigned int vec_x2 = 0, vec_nom = 0, vec_denom = 0, vec_c0 = 0, vec_c1 = 0, vec_c2 = 0, vec_c3 = 0, vec_c1_d = 0, vec_c2_d = 0, vec_c3_d = 0, vec_hi_bound = 0, vec_lo_bound = 0, vec_ones = 0, vec_neg_ones = 0, vec_halves = 0, mask_hi = 0, mask_lo = 0;
  unsigned int acvt_flags = (i_mateltwise_desc->operation == (unsigned char)LIBXSMM_MELTW_OPERATION_ACT_CVTFP32BF16) ? (unsigned int)i_mateltwise_desc->flags : (unsigned int)LIBXSMM_MELTW_FLAG_ACVT_NONE;
  unsigned int cvta_flags = (i_mateltwise_desc->operation == (unsigned char)LIBXSMM_MELTW_OPERATION_CVTFP32BF16_ACT) ? (unsigned int)i_mateltwise_desc->flags : (unsigned int)LIBXSMM_MELTW_FLAG_CVTA_NONE;
  unsigned int fuse_tanh_before_cvt = ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_ACT_CVTFP32BF16) && ((acvt_flags & LIBXSMM_MELTW_FLAG_ACVT_FUSE_TANH) > 0) ) ? 1 : 0;
  unsigned int fuse_sigmoid_before_cvt = ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_ACT_CVTFP32BF16) &&  ((acvt_flags & LIBXSMM_MELTW_FLAG_ACVT_FUSE_SIGM) > 0) ) ? 1 : 0;
  unsigned int fuse_relu_after_cvt = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16_ACT) && ((cvta_flags & LIBXSMM_MELTW_FLAG_CVTA_FUSE_RELU) > 0) ) ? 1 : 0;
  unsigned int fuse_tanh_after_cvt = ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16_ACT) && ((cvta_flags & LIBXSMM_MELTW_FLAG_CVTA_FUSE_TANH) > 0) ) ? 1 : 0;
  unsigned int fuse_sigmoid_after_cvt = ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16_ACT) &&  ((cvta_flags & LIBXSMM_MELTW_FLAG_CVTA_FUSE_SIGM) > 0) ) ? 1 : 0;

  if ((fuse_tanh_before_cvt == 1) && (fuse_sigmoid_before_cvt == 1)) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* For now the below options are not supported in JITer */
  if ((fuse_tanh_after_cvt == 1) || (fuse_sigmoid_after_cvt == 1)) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Some rudimentary checking of M, N and LDs*/
  if ( (i_mateltwise_desc->m > i_mateltwise_desc->ldi) || (i_mateltwise_desc->m > i_mateltwise_desc->ldo) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R11;
  if ( fuse_relu_after_cvt == 1 ) {
    i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_X86_GP_REG_R13;
  }

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

  if ( fuse_relu_after_cvt == 1 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        16,
        i_gp_reg_mapping->gp_reg_relumask,
        0 );
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
    reserved_zmms += 3;
    reserved_mask_regs += 2;
  }

  /* Determine the names of the reserved registers and load with constants when applicable... */
  if ( (fuse_tanh_before_cvt == 1) || (fuse_sigmoid_before_cvt == 1) ) {
    float c0_array[16] = { 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f, 2027025.0f };
    float c1_array[16] = { 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f, 270270.0f };
    float c2_array[16] = { 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f, 6930.0f };
    float c3_array[16] = { 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f, 36.0f };
    float c1_d_array[16] = { 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f, 945945.0f };
    float c2_d_array[16] = { 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f, 51975.0f };
    float c3_d_array[16] = { 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f, 630.0f };
    float hi_b_array[16] = { 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f, 4.97f };
    float lo_b_array[16] = { -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f, -4.97f };
    float ones_array[16] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    float neg_ones_array[16] = { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f };

    reserved_zmms += 14;
    if (fuse_sigmoid_before_cvt == 1) {
      reserved_zmms++;
    }
    vec_x2        = reserved_zmms - 1;
    vec_nom       = reserved_zmms - 2;
    vec_denom     = reserved_zmms - 3;
    vec_c0        = reserved_zmms - 4;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c0_array, "c0_array_", i_micro_kernel_config->vector_name, vec_c0);
    vec_c1        = reserved_zmms - 5;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_array, "c1_array_", i_micro_kernel_config->vector_name, vec_c1);
    vec_c2        = reserved_zmms - 6;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_array, "c2_array_", i_micro_kernel_config->vector_name, vec_c2);
    vec_c3        = reserved_zmms - 7;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c3_array, "c3_array_", i_micro_kernel_config->vector_name, vec_c3);
    vec_c1_d      = reserved_zmms - 8;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_d_array, "c1_d_array_", i_micro_kernel_config->vector_name, vec_c1_d);
    vec_c2_d      = reserved_zmms - 9;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_d_array, "c2_d_array_", i_micro_kernel_config->vector_name, vec_c2_d);
    vec_c3_d      = reserved_zmms - 10;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c3_d_array, "c3_d_array_", i_micro_kernel_config->vector_name, vec_c3_d);
    vec_hi_bound  = reserved_zmms - 11;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) hi_b_array, "hi_b_array_", i_micro_kernel_config->vector_name, vec_hi_bound);
    vec_lo_bound  = reserved_zmms - 12;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) lo_b_array, "lo_b_array_", i_micro_kernel_config->vector_name, vec_lo_bound);
    vec_ones      = reserved_zmms - 13;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) ones_array, "ones_array_", i_micro_kernel_config->vector_name, vec_ones);
    vec_neg_ones  = reserved_zmms - 14;
    libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) neg_ones_array, "neg_ones_array_", i_micro_kernel_config->vector_name, vec_neg_ones);
    if (fuse_sigmoid_before_cvt == 1) {
      float halves_array[16] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
      vec_halves  = reserved_zmms - 15;
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) halves_array, "halves_array_", i_micro_kernel_config->vector_name, vec_halves);
    }
  }

  /* Set zero register neede for relu  */
  if ( fuse_relu_after_cvt == 1 ) {
    reserved_zmms++;
    zero_reg = reserved_zmms - 1;
    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VPXORD,
                                             i_micro_kernel_config->vector_name,
                                             zero_reg, zero_reg, zero_reg );
  }

  /* We fully unroll in M dimension, calculate mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % 32 == 0 ) ? 0 : 1;
  m_trips           = (m + 31) / 32;

  max_nm_unrolling = max_nm_unrolling - reserved_zmms;

  if (m_trips > max_nm_unrolling) {
    eager_result_store = 1;
    n_unroll_factor = 1;
  } else {
    /* Explore n unrolling opportunities... We unroll only by factors that divide N  */
    eager_result_store = 0;
    n_unroll_factor = n;
    while (m_trips * n_unroll_factor > max_nm_unrolling) {
      n_unroll_factor--;
    }
    while (n % n_unroll_factor > 0) {
      n_unroll_factor--;
    }
  }
  n_trips = n / n_unroll_factor;

  /* Calculate input and output masks in case we see m_masking */
  if (use_m_masking == 1) {
    /* If the remaining elements are < 16, then we read a full vector and a partial one at the last m trip */
    /* If the remaining elements are >= 16, then we read a partial vector at the last m trip  */
    /* Calculate mask reg 1 for input-reading */
    mask_in_count = ( (m % 32) > 16) ? 32 - (m % 32) : 16 - (m % 32);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R12, 1, mask_in_count, LIBXSMM_GEMM_PRECISION_F32);
    /* Calculate mask reg 2 for output-writing */
    mask_out_count = 32 - (m % 32);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R12, 2, mask_out_count, LIBXSMM_GEMM_PRECISION_BF16);
    reserved_mask_regs += 2;
  }

  /* Determine the names of the reserved registers... */
  if ( (fuse_tanh_before_cvt == 1) || (fuse_sigmoid_before_cvt == 1) ) {
    reserved_mask_regs += 2;
    mask_hi            = reserved_mask_regs - 1;
    mask_lo            = reserved_mask_regs - 2;
  }

  /* In this case we have to use CPX replacement sequence for downconverts... */
  if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
    /* init stack with helper variables for SW-based RNE rounding */
    /* push 0x7f800000 on the stack, naninf masking */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R12, 0x7f800000);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );

    /* push 0x00010000 on the stack, fixup masking */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R12, 0x00010000);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );

    /* push 0x00007fff on the stack, rneadd */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R12, 0x00007fff);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12);

    /* push 0x00000001 on the stack, fixup */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R12, 0x00000001);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );

    /* If we are using the 3 operant convert variant, then generate the proper permute table in zmm2 for the replacement code */
    if (m > 16) {
      short perm_array[32] = { 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
      short selector_array[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
      for (i = 0; i < 32; i++) {
        perm_array[i] = (short)(perm_array[i] | selector_array[i]);
      }
      libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code,
          (const unsigned char *) perm_array,
          "perm_arrray_",
          i_micro_kernel_config->vector_name,
          2);
    }
  }

  if (n_trips > 1) {
    /* open n loop */
    libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
  }

  for (in = 0; in < n_unroll_factor; in++) {
    for (im = 0; im < m_trips; im++) {
      unroll_iter = in * m_trips + im;
      if (unroll_iter + reserved_zmms < 16) {
        reg_0 = unroll_iter % (16-reserved_zmms) + reserved_zmms;
        reg_1 = unroll_iter % (16-reserved_zmms) + reserved_zmms + 16;
      } else {
        reg_0 = 16 + ((unroll_iter-16+reserved_zmms) % 15);
        reg_1 = reg_0 + 1;
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * 32 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          reg_0, ((im == (m_trips-1)) && (m % 32 < 16)) ? use_m_masking : 0, 1, 0 );

      /* If last iteration and remainder is less than 16, do not load anything  */
      if (!((use_m_masking == 1) && (im == m_trips-1) && (m % 32 <= 16))) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + 16 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            reg_1, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
      }

      if ( (fuse_tanh_before_cvt == 1) || (fuse_sigmoid_before_cvt == 1) ) {
        if (fuse_sigmoid_before_cvt == 1) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                reg_0, vec_halves, reg_0 );
        }

        libxsmm_generator_tanh_ps_rational_78_avx512(io_generated_code, i_micro_kernel_config, reg_0, vec_x2, vec_nom, vec_denom, mask_hi, mask_lo,
                                                     vec_c0, vec_c1, vec_c2, vec_c3, vec_c1_d, vec_c2_d, vec_c3_d, vec_hi_bound, vec_lo_bound, vec_ones, vec_neg_ones);

        if (fuse_sigmoid_before_cvt == 1) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VADDPS,
                                                i_micro_kernel_config->vector_name,
                                                reg_0, vec_ones, reg_0 );

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                reg_0, vec_halves, reg_0 );
        }

        if (!((use_m_masking == 1) && (im == m_trips-1) && (m % 32 <= 16))) {
          if (fuse_sigmoid_before_cvt == 1) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                  i_micro_kernel_config->instruction_set,
                                                  LIBXSMM_X86_INSTR_VMULPS,
                                                  i_micro_kernel_config->vector_name,
                                                  reg_1, vec_halves, reg_1 );
          }

          libxsmm_generator_tanh_ps_rational_78_avx512(io_generated_code, i_micro_kernel_config, reg_1, vec_x2, vec_nom, vec_denom, mask_hi, mask_lo,
                                                       vec_c0, vec_c1, vec_c2, vec_c3, vec_c1_d, vec_c2_d, vec_c3_d, vec_hi_bound, vec_lo_bound, vec_ones, vec_neg_ones);

          if (fuse_sigmoid_before_cvt == 1) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                  i_micro_kernel_config->instruction_set,
                                                  LIBXSMM_X86_INSTR_VADDPS,
                                                  i_micro_kernel_config->vector_name,
                                                  reg_1, vec_ones, reg_1 );

            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                  i_micro_kernel_config->instruction_set,
                                                  LIBXSMM_X86_INSTR_VMULPS,
                                                  i_micro_kernel_config->vector_name,
                                                  reg_1, vec_halves, reg_1 );

          }
        }
      }

      /* Downconvert to BF16  */
      if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) {
        if (!((use_m_masking == 1) && (im == m_trips-1) && (m % 32 <= 16))) {
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
              i_micro_kernel_config->vector_name,
              reg_0, reg_1,
              reg_0,
              0);
        } else {
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
              i_micro_kernel_config->vector_name,
              reg_0, LIBXSMM_X86_VEC_REG_UNDEF,
              reg_0,
              0);
        }
      } else {
        if (!((use_m_masking == 1) && (im == m_trips-1) && (m % 32 <= 16))) {
          /* RNE convert reg_0 and reg_1 */
          libxsmm_generator_cvtfp32bf16_avx512_replacement_sequence( io_generated_code, i_micro_kernel_config, reg_0, 0, 1 );
          libxsmm_generator_cvtfp32bf16_avx512_replacement_sequence( io_generated_code, i_micro_kernel_config, reg_1, 0, 1 );
          /* Properly interleave reg_0 and reg_1 into reg_0  */
          libxsmm_x86_instruction_vec_compute_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPERMT2W,
              i_micro_kernel_config->vector_name,
              reg_1,
              2,
              reg_0);
        } else {
          /* RNE convert reg_0 */
          libxsmm_generator_cvtfp32bf16_avx512_replacement_sequence( io_generated_code, i_micro_kernel_config, reg_0, 0, 1 );
          /* shift FP32 by 16bit to right */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSRAD_I,
              i_micro_kernel_config->vector_name,
              reg_0,
              LIBXSMM_X86_VEC_REG_UNDEF,
              reg_0,
              16);
          /* store 16 bit values into lower portion of reg_0 */
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPMOVDW,
              i_micro_kernel_config->vector_name,
              reg_0, LIBXSMM_X86_VEC_REG_UNDEF,
              reg_0,
              LIBXSMM_X86_VEC_REG_UNDEF);
        }
      }

      if ( fuse_relu_after_cvt == 1 ) {
        /* Compute relu mask */
        if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) {
          current_mask_reg = reserved_mask_regs + (unroll_iter % (8-reserved_mask_regs));
        } else {
          current_mask_reg = reserved_mask_regs - 2 + (unroll_iter % (8-reserved_mask_regs));
        }
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPCMPW,
            i_micro_kernel_config->vector_name,
            zero_reg,
            reg_0,
            reg_0,
            6, current_mask_reg, 0 );

        /* Store relu mask */
        libxsmm_x86_instruction_mask_move_mem( io_generated_code,
            LIBXSMM_X86_INSTR_KMOVD_ST,
            i_gp_reg_mapping->gp_reg_relumask,
            LIBXSMM_X86_GP_REG_UNDEF,
            0,
            (im * 32 + in * i_mateltwise_desc->ldo)/8,
            current_mask_reg );

        /* Blend output result with zero reg based on relu mask */
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPBLENDMW,
            i_micro_kernel_config->vector_name,
            reg_0,
            zero_reg,
            reg_0,
            LIBXSMM_X86_IMM_UNDEF,
            current_mask_reg,
            0 );
      }

      /* Store the result here if have "eager store" decision */
      if (eager_result_store == 1) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            reg_0, (im == (m_trips-1)) ? use_m_masking * 2 : 0, 0, 1 );
      }
    }
  }

  /* Store computed results... */
  if (eager_result_store == 0) {
    for (in = 0; in < n_unroll_factor; in++) {
      for (im = 0; im < m_trips; im++) {
        unroll_iter = in * m_trips + im;
        if (unroll_iter + reserved_zmms < 16) {
          reg_0 = unroll_iter % (16-reserved_zmms) + reserved_zmms;
        } else {
          reg_0 = 16 + ((unroll_iter-16+reserved_zmms) % 15);
        }
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            reg_0, (im == (m_trips-1)) ? use_m_masking * 2 : 0, 0, 1 );
      }
    }
  }

  if (n_trips > 1) {
    /* Adjust input and output pointer */
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_in,
        i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);

    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_out,
        i_mateltwise_desc->ldo *  n_unroll_factor * i_micro_kernel_config->datatype_size_out);

    /* In case of fused relu adjust also relu ptr, datatype for relumask tensor is "bit" and also it has always the same shape as output  */
    if ( fuse_relu_after_cvt == 1 ) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_relumask,
          (i_mateltwise_desc->ldo * n_unroll_factor)/8);
    }

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_ncnc_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int bn, bc, N, C, Nb, iM, im, in, vreg0, vreg1;
  unsigned int use_m_masking, m_trips, m_outer_trips, m_inner_trips, mask_in_count, mask_out_count;
  unsigned int cur_acc0, cur_acc1, mask_load_0, mask_load_1, mask_store;

  bc  = i_mateltwise_desc->m;
  bn  = i_mateltwise_desc->n;
  C   = i_mateltwise_desc->ldi;
  N   = i_mateltwise_desc->ldo;

  Nb  = N/bn;

  if ( (N % bn != 0)  || (C % bc != 0) ) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R10;

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

  use_m_masking     = (bc % 32 == 0) ? 0 : 1;
  m_trips           = (bc + 31)/32;
  m_outer_trips     = (m_trips + 7)/8;

  /* Calculate input mask in case we see m_masking */
  if (use_m_masking == 1) {
    /* If the remaining elements are < 16, then we read a full vector and a partial one at the last m trip */
    /* If the remaining elements are >= 16, then we read a partial vector at the last m trip  */
    /* Calculate mask reg 1 for input-reading */
    mask_in_count = ( (bc % 32) > 16) ? 32 - (bc % 32) : 16 - (bc % 32);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 1, mask_in_count, LIBXSMM_GEMM_PRECISION_F32);
    /* Calculate mask reg 2 for output-writing */
    if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
      mask_out_count = 32 - (bc % 32);
      libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 2, mask_out_count, LIBXSMM_GEMM_PRECISION_BF16);
    } else {
      mask_out_count = 16 - (bc % 16);
      libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 2, mask_out_count, LIBXSMM_GEMM_PRECISION_F32);
    }
  }

  /* Register allocation: Registers zmm16-zmm31 are accumulators, zmm0-zmm15 are used for loading input */
  for (iM = 0; iM < m_outer_trips; iM++) {
    m_inner_trips = (iM == m_outer_trips - 1) ? m_trips - iM * 8 : 8;
    for (im = 0; im < m_inner_trips; im++) {
      cur_acc0 = 16 + im * 2;
      cur_acc1 = 16 + im * 2 + 1;
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPXORD,
                                               i_micro_kernel_config->vector_name,
                                               cur_acc0, cur_acc0, cur_acc0 );
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPXORD,
                                               i_micro_kernel_config->vector_name,
                                               cur_acc1, cur_acc1, cur_acc1 );
    }

    if (Nb > 1) {
      /* open n loop */
      libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
    }

    for (in = 0; in < bn; in++ ) {
      for (im = 0; im < m_inner_trips; im++) {
        cur_acc0 = 16 + im * 2;
        cur_acc1 = 16 + im * 2 + 1;
        vreg0    = im * 2;
        vreg1    = im * 2 + 1;
        mask_load_0 = ((use_m_masking == 1) && (bc % 32 < 16) && (iM == m_outer_trips-1) && (im == m_inner_trips - 1)) ? 1 : 0;
        mask_load_1 = ((mask_load_0 == 0) && (use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips - 1)) ? 1 : 0;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (in * bc + im * 32 + iM * 32 * 8) * i_micro_kernel_config->datatype_size_in,
            'y',
            vreg0, mask_load_0, 1, 0 );

        /* convert 16 bit values into 32 bit (integer convert) */
        libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPMOVSXWD,
            i_micro_kernel_config->vector_name,
            vreg0, LIBXSMM_X86_VEC_REG_UNDEF,
            vreg0,
            LIBXSMM_X86_VEC_REG_UNDEF);

        /* shift 16 bits to the left to generate valid FP32 numbers */
        libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPSLLD_I,
            i_micro_kernel_config->vector_name,
            vreg0,
            LIBXSMM_X86_VEC_REG_UNDEF,
            vreg0,
            16);

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             vreg0, cur_acc0, cur_acc0 );

        if (mask_load_0 == 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (in * bc + im * 32 + 16 + iM * 32 * 8) * i_micro_kernel_config->datatype_size_in,
              'y',
              vreg1, mask_load_1, 1, 0 );

          /* convert 16 bit values into 32 bit (integer convert) */
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              vreg1, LIBXSMM_X86_VEC_REG_UNDEF,
              vreg1,
              LIBXSMM_X86_VEC_REG_UNDEF);

          /* shift 16 bits to the left to generate valid FP32 numbers */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              vreg1,
              LIBXSMM_X86_VEC_REG_UNDEF,
              vreg1,
              16);

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               vreg1, cur_acc1, cur_acc1 );
        }
      }
    }

    if (Nb > 1) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          C * bn * i_micro_kernel_config->datatype_size_in);

      /* close n loop */
      libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, Nb);

      /* Readjust reg_in */
      if (m_outer_trips > 1) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            LIBXSMM_X86_INSTR_SUBQ,
            i_gp_reg_mapping->gp_reg_in,
            C * N * i_micro_kernel_config->datatype_size_in);
      }
    }

    for (im = 0; im < m_inner_trips; im++) {
      cur_acc0 = 16 + im * 2;
      cur_acc1 = 16 + im * 2 + 1;
      mask_store = ((use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips - 1)) ? 1 : 0;

      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
        libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VCVTNE2PS2BF16,
            i_micro_kernel_config->vector_name,
            cur_acc0, cur_acc1,
            cur_acc0,
            0);

        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_out,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (im * 32 + iM * 32 * 8) * i_micro_kernel_config->datatype_size_out,
                                          i_micro_kernel_config->vector_name,
                                          cur_acc0, mask_store * 2, 0, 1 );
      } else {
        mask_load_0 = ((use_m_masking == 1) && (bc % 32 < 16) && (iM == m_outer_trips-1) && (im == m_inner_trips - 1)) ? 1 : 0;
        mask_load_1 = ((mask_load_0 == 0) && (use_m_masking == 1) && (iM == m_outer_trips-1) && (im == m_inner_trips - 1)) ? 1 : 0;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VMOVUPS,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + iM * 32 * 8) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            cur_acc0, mask_load_0 * 2, 0, 1 );

        if (mask_load_0 == 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VMOVUPS,
              i_gp_reg_mapping->gp_reg_out,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * 32 + iM * 32 * 8 + 16) * i_micro_kernel_config->datatype_size_out,
              i_micro_kernel_config->vector_name,
              cur_acc1, mask_load_1 * 2, 0, 1 );
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int in, m, n, m_full_trips, use_m_masking, mask_count, compute_squared_vals_reduce, compute_plain_vals_reduce, reg_n, reg_sum = 30, reg_sum_squared = 31;

  /* Some rudimentary checking of M, N and LDs*/
  if ( i_mateltwise_desc->m > i_mateltwise_desc->ldi ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ELTS) > 0 ) {
    compute_plain_vals_reduce= 1;
  } else {
    compute_plain_vals_reduce= 0;
  }

  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ELTS_SQUARED) > 0 ) {
    compute_squared_vals_reduce = 1;
  } else {
    compute_squared_vals_reduce = 0;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in                   = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_reduced_elts         = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_reduced_elts_squared = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_m_loop               = LIBXSMM_X86_GP_REG_R11;
  i_gp_reg_mapping->gp_reg_n_loop               = LIBXSMM_X86_GP_REG_RAX;

  /* load the input pointer and output pointer */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  if ( compute_plain_vals_reduce > 0 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       8,
       i_gp_reg_mapping->gp_reg_reduced_elts,
       0 );
  }

  if ( compute_squared_vals_reduce > 0 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       16,
       i_gp_reg_mapping->gp_reg_reduced_elts_squared,
       0 );
  }

  /* We fully unroll in N dimension, calculate m-mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % 16 == 0 ) ? 0 : 1;
  m_full_trips      = m / 16;

  /* Calculate input mask in case we see m_masking */
  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for input-reading */
    mask_count =  16 - (m % 16);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 1, mask_count, LIBXSMM_GEMM_PRECISION_F32);
  }

  if ( m_full_trips >= 1 ) {
    if (m_full_trips > 1) {
      /* open m loop */
      libxsmm_generator_mateltwise_header_m_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop );
    }

    /* Initialize accumulators to zero */
    if ( compute_plain_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPXORD,
                                               i_micro_kernel_config->vector_name,
                                               reg_sum, reg_sum, reg_sum );
    }

    if ( compute_squared_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPXORD,
                                               i_micro_kernel_config->vector_name,
                                               reg_sum_squared, reg_sum_squared, reg_sum_squared );
    }

    for (in = 0; in < n; in++) {
      reg_n = in % 30;

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          in * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          reg_n, 0, 1, 0 );

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             reg_n, reg_sum, reg_sum );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VFMADD231PS,
                                             i_micro_kernel_config->vector_name,
                                             reg_n, reg_n, reg_sum_squared );
      }
    }

    /* Store computed results */
    if ( compute_plain_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_mapping->gp_reg_reduced_elts,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        0,
                                        i_micro_kernel_config->vector_name,
                                        reg_sum, 0, 0, 1 );

    }

    if ( compute_squared_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        0,
                                        i_micro_kernel_config->vector_name,
                                        reg_sum_squared, 0, 0, 1 );
    }

    if ((m_full_trips > 1) || (use_m_masking == 1)) {
      /* Adjust input and output pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          16 * i_micro_kernel_config->datatype_size_in);

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_reduced_elts,
            16 * i_micro_kernel_config->datatype_size_out);
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
            16 * i_micro_kernel_config->datatype_size_out);
      }
    }

    if (m_full_trips > 1) {
      /* close m loop */
      libxsmm_generator_mateltwise_footer_m_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop, m_full_trips);
    }
  }

  if (use_m_masking == 1) {
    /* Initialize accumulators to zero */
    if ( compute_plain_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPXORD,
                                               i_micro_kernel_config->vector_name,
                                               reg_sum, reg_sum, reg_sum );
    }

    if ( compute_squared_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPXORD,
                                               i_micro_kernel_config->vector_name,
                                               reg_sum_squared, reg_sum_squared, reg_sum_squared );
    }

    for (in = 0; in < n; in++) {
      reg_n = in % 30;

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          in * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          reg_n, use_m_masking, 1, 0 );

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             reg_n, reg_sum, reg_sum );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VFMADD231PS,
                                             i_micro_kernel_config->vector_name,
                                             reg_n, reg_n, reg_sum_squared );
      }
    }

    /* Store computed results */
    if ( compute_plain_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_mapping->gp_reg_reduced_elts,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        0,
                                        i_micro_kernel_config->vector_name,
                                        reg_sum, use_m_masking, 0, 1 );

    }

    if ( compute_squared_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        0,
                                        i_micro_kernel_config->vector_name,
                                        reg_sum_squared, use_m_masking, 0, 1 );
    }
  }
}




LIBXSMM_API_INTERN
void libxsmm_generator_reduce_rows_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int i = 0, im, m, n, m_trips, n_trips, n_full_trips, use_m_masking, use_n_masking, mask_in_count, mask_out_count, n_cols_load = 16, compute_squared_vals_reduce, compute_plain_vals_reduce;

  /* Some rudimentary checking of M, N and LDs*/
  if ( i_mateltwise_desc->m > i_mateltwise_desc->ldi ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ELTS) > 0 ) {
    compute_plain_vals_reduce= 1;
  } else {
    compute_plain_vals_reduce= 0;
  }

  if ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ELTS_SQUARED) > 0 ) {
    compute_squared_vals_reduce = 1;
  } else {
    compute_squared_vals_reduce = 0;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in                   = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_reduced_elts         = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_reduced_elts_squared = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_m_loop               = LIBXSMM_X86_GP_REG_R11;
  i_gp_reg_mapping->gp_reg_n_loop               = LIBXSMM_X86_GP_REG_RAX;

  /* load the input pointer and output pointer */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  if ( compute_plain_vals_reduce > 0 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       8,
       i_gp_reg_mapping->gp_reg_reduced_elts,
       0 );
  }

  if ( compute_squared_vals_reduce > 0 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       16,
       i_gp_reg_mapping->gp_reg_reduced_elts_squared,
       0 );
  }

  /* We fully unroll in M dimension, calculate mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % 16 == 0 ) ? 0 : 1;
  use_n_masking     = ( n % 16 == 0 ) ? 0 : 1;
  m_trips           = (m + 15) / 16;
  n_trips           = (n + 15) / 16;
  n_full_trips      = ( n % 16 == 0 ) ? n_trips : n_trips-1;

  /* Calculate input mask in case we see m_masking */
  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for input-reading */
    mask_in_count =  16 - (m % 16);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 1, mask_in_count, LIBXSMM_GEMM_PRECISION_F32);
  }

  /* Calculate output mask in case we see n_masking */
  if (use_n_masking == 1) {
    /* Calculate mask reg 2 for output-writing */
    mask_out_count = 16 - (n % 16);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 2, mask_out_count, LIBXSMM_GEMM_PRECISION_F32);
  }

  /* move blend mask value to GP register and to mask register 7 */
  if (n != 1) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, 0xff00 );
    libxsmm_x86_instruction_mask_move( io_generated_code, LIBXSMM_X86_INSTR_KMOVW_GPR_LD, LIBXSMM_X86_GP_REG_RAX, 7 );
  }

  if (n_full_trips >= 1) {
    if (n_full_trips > 1) {
      /* open n loop */
      libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
    }

    /* We fully unroll M loop here...  */
    for (im = 0; im < m_trips; im++) {
      /* load 16 columns of input matrix */
      for (i = 0 ; i < 16; i++) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 16 + i * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            i, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
      }

      /* 1st stage */
      /* zmm0/zmm4; 4444 4444 4444 4444 / 0000 0000 0000 0000 -> zmm0: 4444 4444 0000 0000 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     4, 0, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             4, 0, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              4, 4, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 0 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 24 );
      }

      /* zmm8/zmm12; cccc cccc cccc cccc / 8888 8888 8888 8888 -> zmm8: cccc cccc 8888 8888 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     12, 8, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             12, 8, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              8, 8, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              12, 12, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 8 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 28 );
      }

      /* zmm1/zmm5; 5555 5555 5555 5555 / 1111 1111 1111 1111 -> zmm1: 5555 5555 1111 1111 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     5, 1, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             5, 1, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              1, 1, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              5, 5, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 1 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 25 );
      }

      /* zmm9/zmm13; dddd dddd dddd dddd / 9999 9999 9999 9999 -> zmm9: dddd dddd 9999 9999 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     13, 9, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             13, 9, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              9, 9, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              13, 13, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 9 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 29 );
      }

      /* zmm2/zmm6; 6666 6666 6666 6666 / 2222 2222 2222 2222 -> zmm2: 6666 6666 2222 2222 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     6, 2, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             6, 2, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              2, 2, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              6, 6, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 2 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 26 );
      }

      /* zmm10/zmm14; eeee eeee eeee eeee / aaaa aaaa aaaa aaaa -> zmm10: eeee eeee aaaa aaaa */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     14, 10, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             14, 10, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              10, 10, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              14, 14, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 10 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 30 );
      }

      /* zmm3/zmm7; 7777 7777 7777 7777 / 3333 3333 3333 3333  -> zmm3: 7777 7777 3333 3333 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     7, 3, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             7, 3, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              3, 3, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              7, 7, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 3 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 27 );
      }

      /* zmm11/zmm15; ffff ffff ffff ffff / bbbb bbbb bbbb bbbb  -> zmm11: ffff ffff bbbb bbbb */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     15, 11, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             15, 11, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              11, 11, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              15, 15, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 11 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 31 );
      }

      /* 2nd stage */
      /* zmm0/zmm8; 4444 4444 0000 0000 / cccc cccc 8888 8888  -> zmm0: cccc 8888 4444 0000 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 28, 24, 20, 0x88 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 28, 24, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 24 );
      }

      /* zmm1/zmm9; 5555 5555 1111 1111 / dddd dddd 9999 9999  -> zmm1: dddd 9999 5555 1111 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 29, 25, 20, 0x88 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 29, 25, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 25 );
      }

      /* zmm2/zmm10; 6666 6666 2222 2222 / eeee eeee aaaa aaaa  -> zmm2: eeee aaaa 6666 2222 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 30, 26, 20, 0x88 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 30, 26, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 26 );
      }

      /* zmm3/zmm11:  7777 7777 3333 3333 / ffff ffff bbbb bbbb  -> zmm3: ffff bbbb 7777 3333 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 31, 27, 20, 0x88 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 31, 27, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 27 );
      }

      /* 3rd stage */
      /* zmm0/zmm1; cccc 8888 4444 0000 / dddd 9999 5555 1111  -> zmm0: ddcc 9988 5544 1100 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 25, 24, 20, 0x44 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 25, 24, 21, 0xee );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 24 );
      }

      /* zmm2/zmm3; eeee aaaa 6666 2222 / ffff bbbb 7777 3333  -> zmm2: ffee bbaa 7766 3322 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 27, 26, 20, 0x44 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 27, 26, 21, 0xee );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 26 );
      }

      /* 4th stage */
      /* zmm0/zmm2; ddcc 9988 5544 1100 / ffee bbaa 7766 3322  -> zmm0: fedc ba98 7654 3210 */
      if ( compute_plain_vals_reduce > 0 ) {
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

        /* Update the running reduction result */
        if (im == 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            i_micro_kernel_config->vector_name,
                                            0, 0, 0, 1 );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_out,
              i_gp_reg_mapping->gp_reg_reduced_elts,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              i_micro_kernel_config->vector_name,
              1, 0, 1, 0 );

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                  i_micro_kernel_config->instruction_set,
                                                  LIBXSMM_X86_INSTR_VADDPS,
                                                  i_micro_kernel_config->vector_name,
                                                  1, 0, 0 );

          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            i_micro_kernel_config->vector_name,
                                            0, 0, 0, 1 );
        }
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 26, 24, 20, 0x88 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 26, 24, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 24 );

        /* Update the running reduction result */
        if (im == 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            i_micro_kernel_config->vector_name,
                                            24, 0, 0, 1 );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_out,
              i_gp_reg_mapping->gp_reg_reduced_elts_squared,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              i_micro_kernel_config->vector_name,
              25, 0, 1, 0 );

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                  i_micro_kernel_config->instruction_set,
                                                  LIBXSMM_X86_INSTR_VADDPS,
                                                  i_micro_kernel_config->vector_name,
                                                  25, 24, 24 );

          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            i_micro_kernel_config->vector_name,
                                            24, 0, 0, 1 );
        }
      }
    }

    if ((n_full_trips >  1) || (n % 16 != 0)) {
      /* Adjust input and output pointer */
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          16 * i_mateltwise_desc->ldi *  i_micro_kernel_config->datatype_size_in);

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_reduced_elts,
          16 * i_micro_kernel_config->datatype_size_out);
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_reduced_elts_squared,
          16 * i_micro_kernel_config->datatype_size_out);
      }
    }

    if (n_full_trips > 1) {
      /* close n loop */
      libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_full_trips);
    }
  }

  /* In this case we load only partial number of columns  */
  n_cols_load = n % 16;
  /* Special case when we reduce as single column  */
  if (n == 1) {
    unsigned int reg_sum = 2, reg_sum_squared = 3;
    unsigned int cur_vreg;
    /* Initialize accumulators to zero */
    if ( compute_plain_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPXORD,
                                               i_micro_kernel_config->vector_name,
                                               reg_sum, reg_sum, reg_sum );
    }

    if ( compute_squared_vals_reduce > 0 ) {
      libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPXORD,
                                               i_micro_kernel_config->vector_name,
                                               reg_sum_squared, reg_sum_squared, reg_sum_squared );
    }

    for (im = 0; im < m_trips; im++) {
      cur_vreg = im % 28 + 4;
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * 16 * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             cur_vreg, reg_sum, reg_sum );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VFMADD231PS,
                                             i_micro_kernel_config->vector_name,
                                             cur_vreg, cur_vreg, reg_sum_squared );
      }
    }

    /* Now last horizontal reduction and store of the result...  */
    if ( compute_plain_vals_reduce > 0 ) {
      libxsmm_generator_haddps_avx512( io_generated_code, i_micro_kernel_config, reg_sum, 0, 1);
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_mapping->gp_reg_reduced_elts,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        0,
                                        i_micro_kernel_config->vector_name,
                                        reg_sum, 2, 0, 1 );
    }

    if ( compute_squared_vals_reduce > 0 ) {
      libxsmm_generator_haddps_avx512( io_generated_code, i_micro_kernel_config, reg_sum_squared, 0, 1);
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        i_micro_kernel_config->vmove_instruction_out,
                                        i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        0,
                                        i_micro_kernel_config->vector_name,
                                        reg_sum_squared, 2, 0, 1 );
    }
  } else if (n_cols_load != 0) {
    /* We fully unroll M loop here...  */
    for (im = 0; im < m_trips; im++) {
      /* load 16 columns of input matrix */
      for (i = 0 ; i < n_cols_load; i++) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 16 + i * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            i, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
      }

      for ( i = n_cols_load; i < 16; i++) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VPXORD,
                                                 i_micro_kernel_config->vector_name,
                                                 i, i, i );
      }

      /* 1st stage */
      /* zmm0/zmm4; 4444 4444 4444 4444 / 0000 0000 0000 0000 -> zmm0: 4444 4444 0000 0000 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     4, 0, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             4, 0, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              0, 0, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              4, 4, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 0 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 24 );
      }

      if (n_cols_load > 7) {
        /* zmm8/zmm12; cccc cccc cccc cccc / 8888 8888 8888 8888 -> zmm8: cccc cccc 8888 8888 */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                       i_micro_kernel_config->instruction_set,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       12, 8, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               12, 8, 17, 0x4e );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                8, 8, 18 );

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                12, 12, 19 );


          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                       i_micro_kernel_config->instruction_set,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               19, 18, 21, 0x4e );
        }

        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               16, 17, 8 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               20, 21, 28 );
        }
      }

      /* zmm1/zmm5; 5555 5555 5555 5555 / 1111 1111 1111 1111 -> zmm1: 5555 5555 1111 1111 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     5, 1, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             5, 1, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              1, 1, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              5, 5, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 1 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 25 );
      }

      if (n_cols_load > 8) {
        /* zmm9/zmm13; dddd dddd dddd dddd / 9999 9999 9999 9999 -> zmm9: dddd dddd 9999 9999 */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                       i_micro_kernel_config->instruction_set,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       13, 9, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               13, 9, 17, 0x4e );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                9, 9, 18 );

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                13, 13, 19 );


          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                       i_micro_kernel_config->instruction_set,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               19, 18, 21, 0x4e );
        }

        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               16, 17, 9 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               20, 21, 29 );
        }
      }

      /* zmm2/zmm6; 6666 6666 6666 6666 / 2222 2222 2222 2222 -> zmm2: 6666 6666 2222 2222 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     6, 2, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             6, 2, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              2, 2, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              6, 6, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 2 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 26 );
      }

      if (n_cols_load > 9) {
        /* zmm10/zmm14; eeee eeee eeee eeee / aaaa aaaa aaaa aaaa -> zmm10: eeee eeee aaaa aaaa */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                       i_micro_kernel_config->instruction_set,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       14, 10, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               14, 10, 17, 0x4e );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                10, 10, 18 );

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                14, 14, 19 );


          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                       i_micro_kernel_config->instruction_set,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               19, 18, 21, 0x4e );
        }

        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               16, 17, 10 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               20, 21, 30 );
        }
      }

      /* zmm3/zmm7; 7777 7777 7777 7777 / 3333 3333 3333 3333  -> zmm3: 7777 7777 3333 3333 */
      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     7, 3, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             7, 3, 17, 0x4e );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              3, 3, 18 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                              i_micro_kernel_config->instruction_set,
                                              LIBXSMM_X86_INSTR_VMULPS,
                                              i_micro_kernel_config->vector_name,
                                              7, 7, 19 );


        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                     i_micro_kernel_config->instruction_set,
                                                     LIBXSMM_X86_INSTR_VBLENDMPS,
                                                     i_micro_kernel_config->vector_name,
                                                     19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VSHUFF64X2,
                                             i_micro_kernel_config->vector_name,
                                             19, 18, 21, 0x4e );
      }

      if ( compute_plain_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             16, 17, 3 );
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VADDPS,
                                             i_micro_kernel_config->vector_name,
                                             20, 21, 27 );
      }

      if (n_cols_load > 10) {
        /* zmm11/zmm15; ffff ffff ffff ffff / bbbb bbbb bbbb bbbb  -> zmm11: ffff ffff bbbb bbbb */
        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                       i_micro_kernel_config->instruction_set,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       15, 11, 16, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               15, 11, 17, 0x4e );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                11, 11, 18 );

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                i_micro_kernel_config->instruction_set,
                                                LIBXSMM_X86_INSTR_VMULPS,
                                                i_micro_kernel_config->vector_name,
                                                15, 15, 19 );


          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
                                                       i_micro_kernel_config->instruction_set,
                                                       LIBXSMM_X86_INSTR_VBLENDMPS,
                                                       i_micro_kernel_config->vector_name,
                                                       19, 18, 20, LIBXSMM_X86_IMM_UNDEF, 7, 0 );

          libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VSHUFF64X2,
                                               i_micro_kernel_config->vector_name,
                                               19, 18, 21, 0x4e );
        }

        if ( compute_plain_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               16, 17, 11 );
        }

        if ( compute_squared_vals_reduce > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               20, 21, 31 );
        }
      }

      /* 2nd stage */
      /* zmm0/zmm8; 4444 4444 0000 0000 / cccc cccc 8888 8888  -> zmm0: cccc 8888 4444 0000 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 28, 24, 20, 0x88 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 28, 24, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 24 );
      }

      /* zmm1/zmm9; 5555 5555 1111 1111 / dddd dddd 9999 9999  -> zmm1: dddd 9999 5555 1111 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 29, 25, 20, 0x88 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 29, 25, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 25 );
      }

      /* zmm2/zmm10; 6666 6666 2222 2222 / eeee eeee aaaa aaaa  -> zmm2: eeee aaaa 6666 2222 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 30, 26, 20, 0x88 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 30, 26, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 26 );
      }

      /* zmm3/zmm11:  7777 7777 3333 3333 / ffff ffff bbbb bbbb  -> zmm3: ffff bbbb 7777 3333 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 31, 27, 20, 0x88 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFF64X2,
                                                 i_micro_kernel_config->vector_name,
                                                 31, 27, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 27 );
      }

      /* 3rd stage */
      /* zmm0/zmm1; cccc 8888 4444 0000 / dddd 9999 5555 1111  -> zmm0: ddcc 9988 5544 1100 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 25, 24, 20, 0x44 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 25, 24, 21, 0xee );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 24 );
      }

      /* zmm2/zmm3; eeee aaaa 6666 2222 / ffff bbbb 7777 3333  -> zmm2: ffee bbaa 7766 3322 */
      if ( compute_plain_vals_reduce > 0 ) {
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
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 27, 26, 20, 0x44 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 27, 26, 21, 0xee );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 26 );
      }

      /* 4th stage */
      /* zmm0/zmm2; ddcc 9988 5544 1100 / ffee bbaa 7766 3322  -> zmm0: fedc ba98 7654 3210 */
      if ( compute_plain_vals_reduce > 0 ) {
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

        /* Update the running reduction result */
        if (im == 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            i_micro_kernel_config->vector_name,
                                            0, 2, 0, 1 );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_out,
              i_gp_reg_mapping->gp_reg_reduced_elts,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              i_micro_kernel_config->vector_name,
              1, 2, 1, 0 );

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                  i_micro_kernel_config->instruction_set,
                                                  LIBXSMM_X86_INSTR_VADDPS,
                                                  i_micro_kernel_config->vector_name,
                                                  1, 0, 0 );

          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            i_micro_kernel_config->vector_name,
                                            0, 2, 0, 1 );
        }
      }

      if ( compute_squared_vals_reduce > 0 ) {
        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 26, 24, 20, 0x88 );

        libxsmm_x86_instruction_vec_shuffle_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VSHUFPS,
                                                 i_micro_kernel_config->vector_name,
                                                 26, 24, 21, 0xdd );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 20, 21, 24 );

        /* Update the running reduction result */
        if (im == 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            i_micro_kernel_config->vector_name,
                                            24, 2, 0, 1 );
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_out,
              i_gp_reg_mapping->gp_reg_reduced_elts_squared,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              0,
              i_micro_kernel_config->vector_name,
              25, 2, 1, 0 );

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                  i_micro_kernel_config->instruction_set,
                                                  LIBXSMM_X86_INSTR_VADDPS,
                                                  i_micro_kernel_config->vector_name,
                                                  25, 24, 24 );

          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            i_micro_kernel_config->instruction_set,
                                            i_micro_kernel_config->vmove_instruction_out,
                                            i_gp_reg_mapping->gp_reg_reduced_elts_squared,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            0,
                                            i_micro_kernel_config->vector_name,
                                            24, 2, 0, 1 );
        }
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_scale_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int in, im, m, n, m_full_trips, m_trips, use_m_masking, mask_count, reg_n, reg_m;
  unsigned int scale_rows = 0, scale_cols = 0, scale_rows_cols = 0, perform_scale = 0, perform_shift = 0, perform_addbias = 0;
  unsigned int reg_shift = 31, reg_bias = 30, reg_scale = 29;
  unsigned int reg_shift2 = 28, reg_bias2 = 27, reg_scale2 = 26;
  unsigned int n_available_zmms = 29;
  unsigned int scale_rows_bcastval_accumulate = 0;

  /* Some rudimentary checking of M, N and LDs*/
  if ( i_mateltwise_desc->m > i_mateltwise_desc->ldi ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* Determine what operations to perform */
  scale_rows_cols = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_ROWS_COLS) > 0) ? 1 : 0;
  scale_rows    = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_ROWS) > 0) ? 1 : 0;
  scale_cols    = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_COLS) > 0) ? 1 : 0;
  perform_scale = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_MULT) > 0) ? 1 : 0;
  perform_shift = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_SHIFT) > 0) ? 1 : 0;
  perform_addbias = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_ADD_BIAS) > 0) ? 1 : 0;
  scale_rows_bcastval_accumulate = ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_ROWS_BCASTVAL_ACCUMULATE) > 0) ? 1 : 0;

  if (((scale_rows > 0) && (scale_cols > 0)) ||
      ((scale_cols > 0) && (scale_rows_cols > 0)) ||
      ((scale_rows > 0) && (scale_rows_cols > 0)) ||
      ((scale_rows == 0) && (scale_cols == 0) && (scale_rows_cols == 0) && (scale_rows_bcastval_accumulate == 0))) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* When we request to JIT for rows-cols order, then honor (for code gen purposes) the scale_rows flag */
  if (scale_rows_cols > 0) {
    scale_rows = 1;
    n_available_zmms = 26;
  }

  if (scale_rows_bcastval_accumulate > 0) {
    n_available_zmms = 30;
    reg_scale = 31;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in                     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out                    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_shift_vals             = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_scale_vals             = LIBXSMM_X86_GP_REG_R11;
  i_gp_reg_mapping->gp_reg_bias_vals              = LIBXSMM_X86_GP_REG_R12;
  i_gp_reg_mapping->gp_reg_shift_vals2            = LIBXSMM_X86_GP_REG_R14;
  i_gp_reg_mapping->gp_reg_scale_vals2            = LIBXSMM_X86_GP_REG_RBX;
  i_gp_reg_mapping->gp_reg_bias_vals2             = LIBXSMM_X86_GP_REG_RCX;
  i_gp_reg_mapping->gp_reg_m_loop                 = LIBXSMM_X86_GP_REG_R13;
  i_gp_reg_mapping->gp_reg_n_loop                 = LIBXSMM_X86_GP_REG_RAX;

  /* We fully unroll in N dimension, calculate m-mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % 16 == 0 ) ? 0 : 1;

  /* Calculate input mask in case we see m_masking */
  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for input-reading */
    mask_count =  16 - (m % 16);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 1, mask_count, LIBXSMM_GEMM_PRECISION_F32);
  }

  /* load the input pointer(s) and output pointer */
  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_in,
      0 );

  if ( perform_shift > 0 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
       i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       8,
       i_gp_reg_mapping->gp_reg_shift_vals,
       0 );
  }

  if ( (perform_scale > 0) || (scale_rows_bcastval_accumulate > 0) ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
       i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       16,
       i_gp_reg_mapping->gp_reg_scale_vals,
       0 );
  }

  if ( perform_addbias > 0 ) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
       i_micro_kernel_config->alu_mov_instruction,
       i_gp_reg_mapping->gp_reg_param_struct,
       LIBXSMM_X86_GP_REG_UNDEF, 0,
       24,
       i_gp_reg_mapping->gp_reg_bias_vals,
       0 );
  }

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      32,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  if (scale_rows_cols > 0) {
    if ( perform_shift > 0 ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
         i_micro_kernel_config->alu_mov_instruction,
         i_gp_reg_mapping->gp_reg_param_struct,
         LIBXSMM_X86_GP_REG_UNDEF, 0,
         40,
         i_gp_reg_mapping->gp_reg_shift_vals2,
         0 );
    }

    if ( perform_scale > 0 ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
         i_micro_kernel_config->alu_mov_instruction,
         i_gp_reg_mapping->gp_reg_param_struct,
         LIBXSMM_X86_GP_REG_UNDEF, 0,
         48,
         i_gp_reg_mapping->gp_reg_scale_vals2,
         0 );
    }

    if ( perform_addbias > 0 ) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
         i_micro_kernel_config->alu_mov_instruction,
         i_gp_reg_mapping->gp_reg_param_struct,
         LIBXSMM_X86_GP_REG_UNDEF, 0,
         56,
         i_gp_reg_mapping->gp_reg_bias_vals2,
         0 );
    }
  }

  /* If scaling cols: follow an MN loop order with fully unrolled N loop */
  if (scale_cols == 1) {
    m_full_trips = m / 16;

    if ( m_full_trips >= 1 ) {
      if (m_full_trips > 1) {
        /* open m loop */
        libxsmm_generator_mateltwise_header_m_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop );
      }

      /* Load the correspodning columns to be used for scaling */
      if ( perform_shift > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_shift_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_shift, 0, 1, 0 );
      }

      if ( perform_scale > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_scale_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_scale, 0, 1, 0 );
      }


      if ( perform_addbias > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_bias_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_bias, 0, 1, 0 );
      }

      for (in = 0; in < n; in++) {
        reg_n = in % n_available_zmms;

        /* Load part of the column  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            in * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            reg_n, 0, 1, 0 );

        /* Perform transformations */
        if ( perform_shift > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_shift, reg_n );
        }

        if ( perform_scale > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VMULPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_scale, reg_n );
        }

        if ( perform_addbias> 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_bias, reg_n );
        }

        /* Store part of the column */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_out,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          in * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                          i_micro_kernel_config->vector_name,
                                          reg_n, 0, 0, 1 );
      }

      if ((m_full_trips > 1) || (use_m_masking == 1)) {
        /* Adjust input and output pointer */
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_in,
            16 * i_micro_kernel_config->datatype_size_in);

        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_out,
            16 * i_micro_kernel_config->datatype_size_out);

        if ( perform_shift > 0 ) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_shift_vals,
              16 * i_micro_kernel_config->datatype_size_in);
        }

        if ( perform_scale > 0 ) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_scale_vals,
              16 * i_micro_kernel_config->datatype_size_in);
        }

        if ( perform_addbias > 0 ) {
          libxsmm_x86_instruction_alu_imm(  io_generated_code,
              i_micro_kernel_config->alu_add_instruction,
              i_gp_reg_mapping->gp_reg_bias_vals,
              16 * i_micro_kernel_config->datatype_size_in);
        }
      }

      if (m_full_trips > 1) {
        /* close m loop */
        libxsmm_generator_mateltwise_footer_m_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop, m_full_trips);
      }
    }

    if (use_m_masking == 1) {
      /* Load the correspodning columns to be used for scaling */
      if ( perform_shift > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_shift_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_shift, 0, 1, 0 );
      }

      if ( perform_scale > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_scale_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_scale, 0, 1, 0 );
      }


      if ( perform_addbias > 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_bias_vals,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            0,
            i_micro_kernel_config->vector_name,
            reg_bias, 0, 1, 0 );
      }

      for (in = 0; in < n; in++) {
        reg_n = in % n_available_zmms;

        /* Load part of the column  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            in * i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            reg_n, use_m_masking, 1, 0 );

        /* Perform transformations */
        if ( perform_shift > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_shift, reg_n );
        }

        if ( perform_scale > 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VMULPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_scale, reg_n );
        }

        if ( perform_addbias> 0 ) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VADDPS,
                                               i_micro_kernel_config->vector_name,
                                               reg_n, reg_bias, reg_n );
        }

        /* Store part of the column */
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          i_micro_kernel_config->vmove_instruction_out,
                                          i_gp_reg_mapping->gp_reg_out,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          in * i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out,
                                          i_micro_kernel_config->vector_name,
                                          reg_n, use_m_masking, 0, 1 );
      }
    }
  }

  /* If scaling rows: follow an NM loop order with fully unrolled M loop */
  if ((scale_rows == 1) || (scale_rows_bcastval_accumulate == 1)) {
    m_trips = (m + 15) / 16;
    if (n > 1) {
      /* open n loop */
      libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
    }

    /* Load the correspodning columns to be used for scaling */
    if ( perform_shift > 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VBROADCASTSS,
          i_gp_reg_mapping->gp_reg_shift_vals,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_micro_kernel_config->vector_name,
          reg_shift, 0, 1, 0 );
    }

    if ( (perform_scale > 0) || (scale_rows_bcastval_accumulate == 1) ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VBROADCASTSS,
          i_gp_reg_mapping->gp_reg_scale_vals,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_micro_kernel_config->vector_name,
          reg_scale, 0, 1, 0 );
    }

    if ( perform_addbias > 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VBROADCASTSS,
          i_gp_reg_mapping->gp_reg_bias_vals,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          i_micro_kernel_config->vector_name,
          reg_bias, 0, 1, 0 );
    }

    for (im = 0; im < m_trips; im++) {
      if (scale_rows_bcastval_accumulate == 1) {
        reg_m = (im*2) % n_available_zmms;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            reg_m, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            reg_m+1, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VFMADD231PS,
                                             i_micro_kernel_config->vector_name,
                                             reg_m, reg_scale, reg_m+1 );

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            reg_m+1, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      } else {
        reg_m = im % n_available_zmms;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            reg_m, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        if (scale_rows_cols > 0) {
          /* Load the correspodning columns to be used for scaling */
          if ( perform_shift > 0 ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmove_instruction_in,
                i_gp_reg_mapping->gp_reg_shift_vals2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * 16 * i_micro_kernel_config->datatype_size_in,
                i_micro_kernel_config->vector_name,
                reg_shift2, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
          }

          if ( perform_scale > 0 ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmove_instruction_in,
                i_gp_reg_mapping->gp_reg_scale_vals2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * 16 * i_micro_kernel_config->datatype_size_in,
                i_micro_kernel_config->vector_name,
                reg_scale2, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
          }

          if ( perform_addbias > 0 ) {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmove_instruction_in,
                i_gp_reg_mapping->gp_reg_bias_vals2,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * 16 * i_micro_kernel_config->datatype_size_in,
                i_micro_kernel_config->vector_name,
                reg_bias2, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );
          }
        }

        /* Perform transformations on the rows*/
        if ((perform_scale > 0) && (perform_addbias > 0) && (perform_shift == 0)) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VFMADD213PS,
                                             i_micro_kernel_config->vector_name,
                                             reg_bias, reg_scale, reg_m );
        } else {
          if ( perform_shift > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 reg_m, reg_shift, reg_m );
          }

          if ( perform_scale > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VMULPS,
                                                 i_micro_kernel_config->vector_name,
                                                 reg_m, reg_scale, reg_m );
          }

          if ( perform_addbias > 0 ) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VADDPS,
                                                 i_micro_kernel_config->vector_name,
                                                 reg_m, reg_bias, reg_m );
          }
        }

        if (scale_rows_cols > 0) {
          /* Perform transformations on the columns */
          if ((perform_scale > 0) && (perform_addbias > 0) && (perform_shift == 0)) {
            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VFMADD213PS,
                                               i_micro_kernel_config->vector_name,
                                               reg_bias2, reg_scale2, reg_m );
          } else {
            if ( perform_shift > 0 ) {
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_micro_kernel_config->instruction_set,
                                                   LIBXSMM_X86_INSTR_VADDPS,
                                                   i_micro_kernel_config->vector_name,
                                                   reg_m, reg_shift2, reg_m );
            }

            if ( perform_scale > 0 ) {
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_micro_kernel_config->instruction_set,
                                                   LIBXSMM_X86_INSTR_VMULPS,
                                                   i_micro_kernel_config->vector_name,
                                                   reg_m, reg_scale2, reg_m );
            }

            if ( perform_addbias > 0 ) {
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                   i_micro_kernel_config->instruction_set,
                                                   LIBXSMM_X86_INSTR_VADDPS,
                                                   i_micro_kernel_config->vector_name,
                                                   reg_m, reg_bias2, reg_m );
            }
          }
        }

        /* Store the result  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            reg_m, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      }
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
          i_mateltwise_desc->ldo * i_micro_kernel_config->datatype_size_out);

      if ( perform_shift > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_shift_vals,
            i_micro_kernel_config->datatype_size_in);
      }

      if ( (perform_scale > 0) || (scale_rows_bcastval_accumulate == 1) ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_scale_vals,
            i_micro_kernel_config->datatype_size_in);
      }

      if ( perform_addbias > 0 ) {
        libxsmm_x86_instruction_alu_imm(  io_generated_code,
            i_micro_kernel_config->alu_add_instruction,
            i_gp_reg_mapping->gp_reg_bias_vals,
            i_micro_kernel_config->datatype_size_in);
      }

      /* close n loop */
      libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n);
    }

  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_relu_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int in, im, m, n, use_m_masking, m_trips, n_unroll_factor, n_trips, mask_out_count = 0, unroll_iter = 0;
  unsigned int reserved_mask_regs = 1, n_available_zmms = 31, n_available_mask_regs = 7, max_nm_unrolling = 16;
  unsigned int zero_vreg = 31, cur_vreg = 0, cur_mask_reg = 0;
  int relu_type = -1;
  unsigned int gpr_mask_regs[8] = {LIBXSMM_X86_GP_REG_R8, LIBXSMM_X86_GP_REG_R9, LIBXSMM_X86_GP_REG_R10, LIBXSMM_X86_GP_REG_R11, LIBXSMM_X86_GP_REG_R12, LIBXSMM_X86_GP_REG_R13, LIBXSMM_X86_GP_REG_R14, LIBXSMM_X86_GP_REG_R15};
  unsigned int aggregate_mask_loads = 1;

  /* Determine what relu (fwd/bwd) to perform */
  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_FWD) > 0) {
    relu_type = 0;
  } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_RELU_BWD) > 0) {
    relu_type = 1;
  } else {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Some rudimentary checking of M, N and LDs*/
  if ( i_mateltwise_desc->m > i_mateltwise_desc->ldi ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_RBX;
#if defined(_WIN32) || defined(__CYGWIN__)
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_X86_GP_REG_RDI;
#else
  i_gp_reg_mapping->gp_reg_m_loop = LIBXSMM_X86_GP_REG_RCX;
#endif
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_RSI;
  i_gp_reg_mapping->gp_reg_relumask = LIBXSMM_X86_GP_REG_RDX;

  /* Set zero register needed for relu  */
  libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                           i_micro_kernel_config->instruction_set,
                                           LIBXSMM_X86_INSTR_VPXORD,
                                           i_micro_kernel_config->vector_name,
                                           zero_vreg, zero_vreg, zero_vreg );

  /* We fully unroll in M dimension, calculate mask if there is remainder */
  m                 = i_mateltwise_desc->m;
  n                 = i_mateltwise_desc->n;
  use_m_masking     = ( m % 32 == 0 ) ? 0 : 1;
  m_trips           = (m + 31) / 32;

  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for reading/output-writing */
    mask_out_count = 32 - (m % 32);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R12, 1, mask_out_count, LIBXSMM_GEMM_PRECISION_BF16);
    reserved_mask_regs++;
    n_available_mask_regs--;
  }

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
      16,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      8,
      i_gp_reg_mapping->gp_reg_relumask,
      0 );

  if (m_trips > max_nm_unrolling) {
    n_unroll_factor = 1;
  } else {
    /* Explore n unrolling opportunities... We unroll only by factors that divide N  */
    n_unroll_factor = n;
    while (m_trips * n_unroll_factor > max_nm_unrolling) {
      n_unroll_factor--;
    }
    while (n % n_unroll_factor > 0) {
      n_unroll_factor--;
    }
  }
  n_trips = n / n_unroll_factor;

  if (((n_unroll_factor*m_trips)%2 != 0) || (i_mateltwise_desc->ldo != m)) {
    aggregate_mask_loads = 0;
  }
  if (relu_type == 0) {
    aggregate_mask_loads = 0;
  }

  if (n_trips > 1) {
    /* open n loop */
    libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
  }

  if (aggregate_mask_loads > 0) {
    /* First load all the masks in the auxiliary gprs  */
    for (unroll_iter = 0; unroll_iter < m_trips * n_unroll_factor; unroll_iter += 2) {
      in = unroll_iter/m_trips;
      im = unroll_iter%m_trips;

      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_relumask,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * 32 + in * i_mateltwise_desc->ldo)/8,
          gpr_mask_regs[unroll_iter/2],
          0 );
    }

    /* Now read the input, apply the relu and store the mask */
    for (in = 0; in < n_unroll_factor; in++) {
      for (im = 0; im < m_trips; im++) {
        unroll_iter = in * m_trips + im;
        cur_vreg = unroll_iter % n_available_zmms;
        cur_mask_reg = reserved_mask_regs + unroll_iter % n_available_mask_regs;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        /* Extract the mask from the proper auxiliry gpr */
        libxsmm_x86_instruction_mask_move( io_generated_code,
            LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
            gpr_mask_regs[unroll_iter/2],
            cur_mask_reg );

        /* Shift current mask gpr in order to br able to extract the next mask  */
        if (unroll_iter % 2 == 0) {
          libxsmm_x86_instruction_alu_imm( io_generated_code,
              LIBXSMM_X86_INSTR_SHRQ,
              gpr_mask_regs[unroll_iter/2],
              32 );
        }

        /* Blend output result with zero reg based on relu mask */
        libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPBLENDMW,
            i_micro_kernel_config->vector_name,
            cur_vreg,
            zero_vreg,
            cur_vreg,
            LIBXSMM_X86_IMM_UNDEF,
            cur_mask_reg,
            0 );

        /* Store result  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      }
    }
  } else {
    for (in = 0; in < n_unroll_factor; in++) {
      for (im = 0; im < m_trips; im++) {
        unroll_iter = in * m_trips + im;
        cur_vreg = unroll_iter % n_available_zmms;
        cur_mask_reg = reserved_mask_regs + unroll_iter % n_available_mask_regs;

        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + in * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 1, 0 );

        if (relu_type == 0) {
          /* Compare to generate mask  */
          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPCMPW,
              i_micro_kernel_config->vector_name,
              zero_vreg,
              cur_vreg,
              cur_vreg,
              6, cur_mask_reg, 0 );

          /* Store mask relu  */
          libxsmm_x86_instruction_mask_move_mem( io_generated_code,
              LIBXSMM_X86_INSTR_KMOVD_ST,
              i_gp_reg_mapping->gp_reg_relumask,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              (im * 32 + in * i_mateltwise_desc->ldo)/8,
              cur_mask_reg );

          /* Blend output result with zero reg based on relu mask */
          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPBLENDMW,
              i_micro_kernel_config->vector_name,
              cur_vreg,
              zero_vreg,
              cur_vreg,
              LIBXSMM_X86_IMM_UNDEF,
              cur_mask_reg,
              0 );
        } else {
          /* Load relu mask */
          libxsmm_x86_instruction_mask_move_mem( io_generated_code,
              LIBXSMM_X86_INSTR_KMOVD_LD,
              i_gp_reg_mapping->gp_reg_relumask,
              LIBXSMM_X86_GP_REG_UNDEF,
              0,
              (im * 32 + in * i_mateltwise_desc->ldo)/8,
              cur_mask_reg );

          /* Blend output result with zero reg based on relu mask */
          libxsmm_x86_instruction_vec_compute_reg_mask( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPBLENDMW,
              i_micro_kernel_config->vector_name,
              cur_vreg,
              zero_vreg,
              cur_vreg,
              LIBXSMM_X86_IMM_UNDEF,
              cur_mask_reg,
              0 );
        }

        /* Store result  */
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_out,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            (im * 32 + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
            i_micro_kernel_config->vector_name,
            cur_vreg, (im == (m_trips-1)) ? use_m_masking : 0, 0, 1 );
      }
    }
  }

  if (n_trips > 1) {
    /* Adjust input and output pointer */
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_in,
        i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);

    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_out,
        i_mateltwise_desc->ldo *  n_unroll_factor * i_micro_kernel_config->datatype_size_out);

    /* Adjust also relu ptr, datatype for relumask tensor is "bit" and also it has always the same shape as output  */
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_relumask,
        (i_mateltwise_desc->ldo * n_unroll_factor)/8);

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_opreduce_vecs_index_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int m, im, use_m_masking, m_trips, max_m_unrolling = 4, m_unroll_factor = 1, m_trips_loop = 0, peeled_m_trips = 0, mask_out_count = 0, vecin_offset = 0, vecidxin_offset = 0, vecout_offset = 0, temp_vreg = 31;
  unsigned int idx_tsize =  i_mateltwise_desc->n;
  unsigned int in_tsize = 4;
  unsigned int ind_alu_mov_instr = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_MOVQ : LIBXSMM_X86_INSTR_MOVL;
  int pf_dist = 4, use_nts = 0, pf_instr = LIBXSMM_X86_INSTR_PREFETCHT1, pf_type = 1, load_acc = 1;
  unsigned int vstore_instr = 0;
  int apply_op = 0, apply_redop = 0, op_order = -1, op_instr = 0, reduceop_instr = 0, scale_op_result = 0, pushpop_scaleop_reg = 0;
  unsigned int NO_PF_LABEL_START = 0;
  unsigned int NO_PF_LABEL_START_2 = 1;
  unsigned int END_LABEL = 2;
  const int LIBXSMM_X86_INSTR_DOTPS = -1;
  unsigned int op_mask = 0, reduceop_mask = 0, op_mask_cntl = 0, reduceop_mask_cntl = 0, op_imm = 0, reduceop_imm = 0;
  unsigned char op_sae_cntl = 0, reduceop_sae_cntl = 0;
  const char *const env_pf_dist = getenv("PF_DIST_OPREDUCE_VECS_IDX");
  const char *const env_pf_type = getenv("PF_TYPE_OPREDUCE_VECS_IDX");
  const char *const env_nts     = getenv("NTS_OPREDUCE_VECS_IDX");
  const char *const env_load_acc= getenv("LOAD_ACCS_OPREDUCE_VECS_IDX");
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

  vecin_offset    = vecout_offset + max_m_unrolling;
  vecidxin_offset = vecin_offset + max_m_unrolling;

  if ( 0 == env_pf_dist ) {
  } else {
    pf_dist = atoi(env_pf_dist);
  }
  if ( 0 == env_pf_type ) {
  } else {
    pf_type = atoi(env_pf_type);
  }
  if ( 0 == env_nts ) {
  } else {
    use_nts = atoi(env_nts);
  }
  if ( 0 == env_load_acc ) {
  } else {
    load_acc = atoi(env_load_acc);
  }

  /* TODO: Add validation checks for various JIT params and error out... */

  if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
    in_tsize = 2;
  } else if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
    in_tsize = 4;
  } else {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  pf_instr      = (pf_type == 2) ? LIBXSMM_X86_INSTR_PREFETCHT1 : LIBXSMM_X86_INSTR_PREFETCHT0 ;
  vstore_instr  = (use_nts == 0) ? i_micro_kernel_config->vmove_instruction_out : LIBXSMM_X86_INSTR_VMOVNTPS;

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_in_base  = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_out      = LIBXSMM_X86_GP_REG_R11;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_X86_GP_REG_RDX;
  i_gp_reg_mapping->gp_reg_in       = LIBXSMM_X86_GP_REG_RSI;
  i_gp_reg_mapping->gp_reg_in_pf    = LIBXSMM_X86_GP_REG_RCX;
  i_gp_reg_mapping->gp_reg_invec    = LIBXSMM_X86_GP_REG_RDI;

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_n,
      0 );

  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_n, 0);
  libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JLE, END_LABEL, &l_jump_label_tracker);

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      8,
      i_gp_reg_mapping->gp_reg_ind_base,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      16,
      i_gp_reg_mapping->gp_reg_in_base,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      32,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_SCALE_OP_RESULT) > 0) {
    scale_op_result = 1;
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_NONE) > 0) {
      i_gp_reg_mapping->gp_reg_scale_base = LIBXSMM_X86_GP_REG_RDI;
    } else if (pf_dist == 0) {
      i_gp_reg_mapping->gp_reg_scale_base = LIBXSMM_X86_GP_REG_RCX;
    } else {
      i_gp_reg_mapping->gp_reg_scale_base = LIBXSMM_X86_GP_REG_R12;
      pushpop_scaleop_reg = 1;
      libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scale_base );
    }

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        40,
        i_gp_reg_mapping->gp_reg_scale_base,
        0 );
  }

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_NONE) == 0) {
    apply_op = 1;
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        24,
        i_gp_reg_mapping->gp_reg_invec,
        0 );
  }

  if (apply_op == 1) {
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIN_VECIDX) > 0) {
      op_order = 1;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN) > 0) {
      op_order = 0;
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }

    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_ADD) > 0) {
      op_instr = LIBXSMM_X86_INSTR_VADDPS;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_SUB) > 0) {
      op_instr = LIBXSMM_X86_INSTR_VSUBPS;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_MUL) > 0) {
      op_instr = LIBXSMM_X86_INSTR_VMULPS;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DIV) > 0) {
      op_instr = LIBXSMM_X86_INSTR_VDIVPS;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DOT) > 0) {
      op_instr = LIBXSMM_X86_INSTR_DOTPS;
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }

  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_NONE) == 0) {
    apply_redop = 1;
    if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM) > 0) {
      reduceop_instr = LIBXSMM_X86_INSTR_VADDPS;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX) > 0) {
      reduceop_instr = LIBXSMM_X86_INSTR_VRANGEPS;
      reduceop_imm = 5;
    } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MIN) > 0) {
      reduceop_instr = LIBXSMM_X86_INSTR_VRANGEPS;
      reduceop_imm = 4;
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  } else {
    vecin_offset    = vecout_offset;
  }

  m                 = i_mateltwise_desc->m;
  use_m_masking     = ( m % 16 == 0 ) ? 0 : 1;
  m_trips           = (m + 15) / 16;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for reading/output-writing */
    mask_out_count = 16 - (m % 16);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 1, mask_out_count, LIBXSMM_GEMM_PRECISION_F32);
  }

  /* In this case we have to use CPX replacement sequence for downconverts... */
  if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
    /* init stack with helper variables for SW-based RNE rounding */
    /* push 0x7f800000 on the stack, naninf masking */
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );

    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R13, 0x7f800000);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );

    /* push 0x00010000 on the stack, fixup masking */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R13, 0x00010000);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );

    /* push 0x00007fff on the stack, rneadd */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R13, 0x00007fff);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13);

    /* push 0x00000001 on the stack, fixup */
    libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R13, 0x00000001);
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
  }

  /* In this case we have to generate a loop for m */
  if (m_unroll_factor > max_m_unrolling) {
    m_unroll_factor = max_m_unrolling;
    m_trips_loop = m_trips/m_unroll_factor;
    peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  } else {
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  }

  if (m_trips_loop > 1) {
    libxsmm_generator_mateltwise_header_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop );
  }

  if (m_trips_loop >= 1) {
    for (im = 0; im < m_unroll_factor; im++) {
      /* Load output for reduction  */
      if (apply_redop == 1) {
        if (load_acc == 0) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VPXORD,
                                                 i_micro_kernel_config->vector_name,
                                                 im + vecout_offset, im + vecout_offset, im + vecout_offset);
        } else {
          char vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ? 'y' : 'z';
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_out,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * 16  * i_micro_kernel_config->datatype_size_out,
              vname,
              im + vecout_offset, 0, 0, 0 );

          if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
            /* convert 16 bit values into 32 bit (integer convert) */
            libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPMOVSXWD,
                i_micro_kernel_config->vector_name,
                im + vecout_offset, LIBXSMM_X86_VEC_REG_UNDEF,
                im + vecout_offset,
                LIBXSMM_X86_VEC_REG_UNDEF);

            /* shift 16 bits to the left to generate valid FP32 numbers */
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPSLLD_I,
                i_micro_kernel_config->vector_name,
                im + vecout_offset,
                LIBXSMM_X86_VEC_REG_UNDEF,
                im + vecout_offset,
                16);
          }
        }
      }
      /* Load input vector in case we have to apply op */
      if (apply_op == 1) {
        char vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ? 'y' : 'z';
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_invec,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16  * i_micro_kernel_config->datatype_size_out,
            vname,
            im + vecin_offset, 0, 0, 0 );

        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
          /* convert 16 bit values into 32 bit (integer convert) */
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              im + vecin_offset, LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecin_offset,
              LIBXSMM_X86_VEC_REG_UNDEF);

          /* shift 16 bits to the left to generate valid FP32 numbers */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              im + vecin_offset,
              LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecin_offset,
              16);
        }
      }
    }

    if (pf_dist > 0) {
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_n_loop, 0);
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
      libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JLE, NO_PF_LABEL_START, &l_jump_label_tracker);

      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mapping->gp_reg_n, pf_dist);
      libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, 1);
      libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, i_gp_reg_mapping->gp_reg_in, 0);
      libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, pf_dist * idx_tsize, i_gp_reg_mapping->gp_reg_in_pf, 0);
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * in_tsize);
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in_pf, i_mateltwise_desc->ldi * in_tsize);
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in);
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in_pf);

      for (im = 0; im < m_unroll_factor; im++) {
        /* First load the indexed vector */
        char vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) ? 'y' : 'z';
        libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * 16  * i_micro_kernel_config->datatype_size_in,
          vname,
          im + vecidxin_offset, 0, 0, 0 );

        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
          /* convert 16 bit values into 32 bit (integer convert) */
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              im + vecidxin_offset, LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecidxin_offset,
              LIBXSMM_X86_VEC_REG_UNDEF);

          /* shift 16 bits to the left to generate valid FP32 numbers */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              im + vecidxin_offset,
              LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecidxin_offset,
              16);
        }

        /* Now apply the OP among the indexed vector and the input vector */
        if (apply_op == 1) {
          if (op_instr == LIBXSMM_X86_INSTR_DOTPS) {
            /* TODO: Add DOT op sequence here  */
          } else {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
                op_instr,
                i_micro_kernel_config->vector_name,
                (op_order == 0)    ? im + vecin_offset : im + vecidxin_offset,
                (op_order == 0)    ? im + vecidxin_offset : im + vecin_offset,
                (apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset,
                op_mask,
                op_mask_cntl,
                op_sae_cntl,
                op_imm);
          }
        }

        if (scale_op_result == 1) {
          if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
            libxsmm_x86_instruction_vec_move(io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPBROADCASTW,
                i_gp_reg_mapping->gp_reg_scale_base,
                i_gp_reg_mapping->gp_reg_n_loop,
                in_tsize,
                0, 'y',
                temp_vreg,
                0,
                0,
                0);

            libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPMOVSXWD,
                i_micro_kernel_config->vector_name,
                temp_vreg, LIBXSMM_X86_VEC_REG_UNDEF,
                temp_vreg,
                LIBXSMM_X86_VEC_REG_UNDEF);

            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPSLLD_I,
                i_micro_kernel_config->vector_name,
                temp_vreg,
                LIBXSMM_X86_VEC_REG_UNDEF,
                temp_vreg,
                16);

            libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
                LIBXSMM_X86_INSTR_VMULPS,
                i_micro_kernel_config->vector_name,
                temp_vreg,
                (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
                (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset);
          } else {
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
                LIBXSMM_X86_INSTR_VMULPS,
                i_micro_kernel_config->vector_name,
                i_gp_reg_mapping->gp_reg_scale_base,
                i_gp_reg_mapping->gp_reg_n_loop,
                4, 0, 1,
                (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
                (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
                0, 0, 0);
          }
        }

        /* Now apply the Reduce OP */
        if (apply_redop == 1) {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
              reduceop_instr,
              i_micro_kernel_config->vector_name,
              im + vecidxin_offset,
              im + vecout_offset,
              im + vecout_offset,
              reduceop_mask,
              reduceop_mask_cntl,
              reduceop_sae_cntl,
              reduceop_imm);
        }

        if ((im * 16 * i_micro_kernel_config->datatype_size_in) % 64 == 0 ) {
          libxsmm_x86_instruction_prefetch(io_generated_code,
              pf_instr,
              i_gp_reg_mapping->gp_reg_in_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * 16 * i_micro_kernel_config->datatype_size_in);
        }
      }

      libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
      /* NO_PF_LABEL_START */
      libxsmm_x86_instruction_register_jump_label(io_generated_code, NO_PF_LABEL_START, &l_jump_label_tracker);
    }

    /* Perform the reductions for all columns */
    libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, (pf_dist > 0) ? 1 : 0);
    libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, i_gp_reg_mapping->gp_reg_in, 0);
    libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * in_tsize);
    libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in);

    for (im = 0; im < m_unroll_factor; im++) {
      /* First load the indexed vector */
      char vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) ? 'y' : 'z';
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * 16  * i_micro_kernel_config->datatype_size_in,
          vname,
          im + vecidxin_offset, 0, 0, 0 );

      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
        /* convert 16 bit values into 32 bit (integer convert) */
        libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPMOVSXWD,
            i_micro_kernel_config->vector_name,
            im + vecidxin_offset, LIBXSMM_X86_VEC_REG_UNDEF,
            im + vecidxin_offset,
            LIBXSMM_X86_VEC_REG_UNDEF);

        /* shift 16 bits to the left to generate valid FP32 numbers */
        libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPSLLD_I,
            i_micro_kernel_config->vector_name,
            im + vecidxin_offset,
            LIBXSMM_X86_VEC_REG_UNDEF,
            im + vecidxin_offset,
            16);
      }

      /* Now apply the OP among the indexed vector and the input vector */
      if (apply_op == 1) {
        if (op_instr == LIBXSMM_X86_INSTR_DOTPS) {
          /* TODO: Add DOT op sequence here  */
        } else {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
              op_instr,
              i_micro_kernel_config->vector_name,
              (op_order == 0)    ? im + vecin_offset : im + vecidxin_offset,
              (op_order == 0)    ? im + vecidxin_offset : im + vecin_offset,
              (apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset,
              op_mask,
              op_mask_cntl,
              op_sae_cntl,
              op_imm);
        }
      }

      if (scale_op_result == 1) {
        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
          libxsmm_x86_instruction_vec_move(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPBROADCASTW,
              i_gp_reg_mapping->gp_reg_scale_base,
              i_gp_reg_mapping->gp_reg_n_loop,
              in_tsize,
              0, 'y',
              temp_vreg,
              0,
              0,
              0);

          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              temp_vreg, LIBXSMM_X86_VEC_REG_UNDEF,
              temp_vreg,
              LIBXSMM_X86_VEC_REG_UNDEF);

          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              temp_vreg,
              LIBXSMM_X86_VEC_REG_UNDEF,
              temp_vreg,
              16);

          libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
              LIBXSMM_X86_INSTR_VMULPS,
              i_micro_kernel_config->vector_name,
              temp_vreg,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset);
        } else {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
              LIBXSMM_X86_INSTR_VMULPS,
              i_micro_kernel_config->vector_name,
              i_gp_reg_mapping->gp_reg_scale_base,
              i_gp_reg_mapping->gp_reg_n_loop,
              4, 0, 1,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
              0, 0, 0);
        }
      }

      /* Now apply the Reduce OP */
      if (apply_redop == 1) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
            reduceop_instr,
            i_micro_kernel_config->vector_name,
            im + vecidxin_offset,
            im + vecout_offset,
            im + vecout_offset,
            reduceop_mask,
            reduceop_mask_cntl,
            reduceop_sae_cntl,
            reduceop_imm);
      }
    }

    libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);

    /* Now store accumulators  */
    for (im = 0; im < m_unroll_factor; im++) {
      char vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) ? 'y' : 'z';
      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
        if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) {
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
              i_micro_kernel_config->vector_name,
              im + vecout_offset, LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecout_offset,
              0);
        } else {
          libxsmm_generator_cvtfp32bf16_avx512_replacement_sequence( io_generated_code, i_micro_kernel_config, im + vecout_offset, 30, 31 );
          /* shift FP32 by 16bit to right */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSRAD_I,
              i_micro_kernel_config->vector_name,
              im + vecout_offset,
              LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecout_offset,
              16);
          /* store 16 bit values into lower portion of reg_0 */
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPMOVDW,
              i_micro_kernel_config->vector_name,
              im + vecout_offset, LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecout_offset,
              LIBXSMM_X86_VEC_REG_UNDEF);
        }
      }
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          vstore_instr,
          i_gp_reg_mapping->gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * 16  * i_micro_kernel_config->datatype_size_out,
          vname,
          im + vecout_offset, 0, 0, 1 );

    }
  }

  if (m_trips_loop > 1) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, m_unroll_factor * 16 * i_micro_kernel_config->datatype_size_out);
    if (apply_op == 1) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_invec, m_unroll_factor * 16 * i_micro_kernel_config->datatype_size_out);
    }
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, m_unroll_factor * 16 * i_micro_kernel_config->datatype_size_in);
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
  }

  if (peeled_m_trips > 0) {
    if (m_trips_loop == 1) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, m_unroll_factor * 16 * i_micro_kernel_config->datatype_size_out);
      if (apply_op == 1) {
        libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_invec, m_unroll_factor * 16 * i_micro_kernel_config->datatype_size_out);
      }
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, m_unroll_factor * 16 * i_micro_kernel_config->datatype_size_in);
    }

    /* Perform the reductions for all columns */
    for (im = 0; im < peeled_m_trips; im++) {
      /* Load output for reduction  */
      if (apply_redop == 1) {
        if (load_acc == 0) {
          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 i_micro_kernel_config->instruction_set,
                                                 LIBXSMM_X86_INSTR_VPXORD,
                                                 i_micro_kernel_config->vector_name,
                                                 im + vecout_offset, im + vecout_offset, im + vecout_offset);
        } else {
          char vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ? 'y' : 'z';
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_out,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * 16  * i_micro_kernel_config->datatype_size_out,
              vname,
              im + vecout_offset, (im == peeled_m_trips-1) ? use_m_masking : 0, 0, 0 );

          if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
            /* convert 16 bit values into 32 bit (integer convert) */
            libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPMOVSXWD,
                i_micro_kernel_config->vector_name,
                im + vecout_offset, LIBXSMM_X86_VEC_REG_UNDEF,
                im + vecout_offset,
                LIBXSMM_X86_VEC_REG_UNDEF);

            /* shift 16 bits to the left to generate valid FP32 numbers */
            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPSLLD_I,
                i_micro_kernel_config->vector_name,
                im + vecout_offset,
                LIBXSMM_X86_VEC_REG_UNDEF,
                im + vecout_offset,
                16);
          }
        }
      }
      /* Load input vector in case we have to apply op */
      if (apply_op == 1) {
        char vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ? 'y' : 'z';
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_invec,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16  * i_micro_kernel_config->datatype_size_out,
            vname,
            im + vecin_offset, (im == peeled_m_trips-1) ? use_m_masking : 0, 0, 0 );

        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
          /* convert 16 bit values into 32 bit (integer convert) */
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              im + vecin_offset, LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecin_offset,
              LIBXSMM_X86_VEC_REG_UNDEF);

          /* shift 16 bits to the left to generate valid FP32 numbers */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              im + vecin_offset,
              LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecin_offset,
              16);
        }
      }
    }

    if (pf_dist > 0) {
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_n_loop, 0);
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
      libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JLE, NO_PF_LABEL_START_2, &l_jump_label_tracker);

      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mapping->gp_reg_n, pf_dist);
      libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, 1);
      libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, i_gp_reg_mapping->gp_reg_in, 0);
      libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, pf_dist * idx_tsize, i_gp_reg_mapping->gp_reg_in_pf, 0);
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * in_tsize);
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in_pf, i_mateltwise_desc->ldi * in_tsize);
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in);
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in_pf);

      for (im = 0; im < peeled_m_trips; im++) {
        /* First load the indexed vector */
        char vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) ? 'y' : 'z';
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16  * i_micro_kernel_config->datatype_size_in,
            vname,
            im + vecidxin_offset, (im == peeled_m_trips-1) ? use_m_masking : 0, ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? 1 : 0, 0 );

        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
          /* convert 16 bit values into 32 bit (integer convert) */
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              im + vecidxin_offset, LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecidxin_offset,
              LIBXSMM_X86_VEC_REG_UNDEF);

          /* shift 16 bits to the left to generate valid FP32 numbers */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              im + vecidxin_offset,
              LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecidxin_offset,
              16);
        }

        /* Now apply the OP among the indexed vector and the input vector */
        if (apply_op == 1) {
          if (op_instr == LIBXSMM_X86_INSTR_DOTPS) {
            /* TODO: Add DOT op sequence here  */
          } else {
            libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
                op_instr,
                i_micro_kernel_config->vector_name,
                (op_order == 0)    ? im + vecin_offset : im + vecidxin_offset,
                (op_order == 0)    ? im + vecidxin_offset : im + vecin_offset,
                (apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset,
                op_mask,
                op_mask_cntl,
                op_sae_cntl,
                op_imm);
          }
        }

        if (scale_op_result == 1) {
          if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
            libxsmm_x86_instruction_vec_move(io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPBROADCASTW,
                i_gp_reg_mapping->gp_reg_scale_base,
                i_gp_reg_mapping->gp_reg_n_loop,
                in_tsize,
                0, 'y',
                temp_vreg,
                0,
                0,
                0);

            libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPMOVSXWD,
                i_micro_kernel_config->vector_name,
                temp_vreg, LIBXSMM_X86_VEC_REG_UNDEF,
                temp_vreg,
                LIBXSMM_X86_VEC_REG_UNDEF);

            libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VPSLLD_I,
                i_micro_kernel_config->vector_name,
                temp_vreg,
                LIBXSMM_X86_VEC_REG_UNDEF,
                temp_vreg,
                16);

            libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
                LIBXSMM_X86_INSTR_VMULPS,
                i_micro_kernel_config->vector_name,
                temp_vreg,
                (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
                (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset);
          } else {
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
                LIBXSMM_X86_INSTR_VMULPS,
                i_micro_kernel_config->vector_name,
                i_gp_reg_mapping->gp_reg_scale_base,
                i_gp_reg_mapping->gp_reg_n_loop,
                4, 0, 1,
                (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
                (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
                0, 0, 0);
          }
        }

        /* Now apply the Reduce OP */
        if (apply_redop == 1) {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
              reduceop_instr,
              i_micro_kernel_config->vector_name,
              im + vecidxin_offset,
              im + vecout_offset,
              im + vecout_offset,
              reduceop_mask,
              reduceop_mask_cntl,
              reduceop_sae_cntl,
              reduceop_imm);
        }

        if ((im * 16 * i_micro_kernel_config->datatype_size_in) % 64 == 0 ) {
          libxsmm_x86_instruction_prefetch(io_generated_code,
              pf_instr,
              i_gp_reg_mapping->gp_reg_in_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * 16 * i_micro_kernel_config->datatype_size_in);
        }
      }

      libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
      /* NO_PF_LABEL_START_2 */
      libxsmm_x86_instruction_register_jump_label(io_generated_code, NO_PF_LABEL_START_2, &l_jump_label_tracker);
    }

    libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, (pf_dist > 0) ? 1 : 0);
    libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, i_gp_reg_mapping->gp_reg_in, 0);
    libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * in_tsize);
    libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in);

    for (im = 0; im < peeled_m_trips; im++) {
      /* First load the indexed vector */
      char vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) ? 'y' : 'z';
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_in,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * 16  * i_micro_kernel_config->datatype_size_in,
          vname,
          im + vecidxin_offset, (im == peeled_m_trips-1) ? use_m_masking : 0, ((im == peeled_m_trips-1) && (use_m_masking > 0)) ? 1 : 0, 0 );

      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) {
        /* convert 16 bit values into 32 bit (integer convert) */
        libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPMOVSXWD,
            i_micro_kernel_config->vector_name,
            im + vecidxin_offset, LIBXSMM_X86_VEC_REG_UNDEF,
            im + vecidxin_offset,
            LIBXSMM_X86_VEC_REG_UNDEF);

        /* shift 16 bits to the left to generate valid FP32 numbers */
        libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPSLLD_I,
            i_micro_kernel_config->vector_name,
            im + vecidxin_offset,
            LIBXSMM_X86_VEC_REG_UNDEF,
            im + vecidxin_offset,
            16);
      }

      /* Now apply the OP among the indexed vector and the input vector */
      if (apply_op == 1) {
        if (op_instr == LIBXSMM_X86_INSTR_DOTPS) {
          /* TODO: Add DOT op sequence here  */
        } else {
          libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
              op_instr,
              i_micro_kernel_config->vector_name,
              (op_order == 0)    ? im + vecin_offset : im + vecidxin_offset,
              (op_order == 0)    ? im + vecidxin_offset : im + vecin_offset,
              (apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset,
              op_mask,
              op_mask_cntl,
              op_sae_cntl,
              op_imm);
        }
      }

      if (scale_op_result == 1) {
        if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
          libxsmm_x86_instruction_vec_move(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPBROADCASTW,
              i_gp_reg_mapping->gp_reg_scale_base,
              i_gp_reg_mapping->gp_reg_n_loop,
              in_tsize,
              0, 'y',
              temp_vreg,
              0,
              0,
              0);

          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPMOVSXWD,
              i_micro_kernel_config->vector_name,
              temp_vreg, LIBXSMM_X86_VEC_REG_UNDEF,
              temp_vreg,
              LIBXSMM_X86_VEC_REG_UNDEF);

          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSLLD_I,
              i_micro_kernel_config->vector_name,
              temp_vreg,
              LIBXSMM_X86_VEC_REG_UNDEF,
              temp_vreg,
              16);

          libxsmm_x86_instruction_vec_compute_3reg(io_generated_code,
              LIBXSMM_X86_INSTR_VMULPS,
              i_micro_kernel_config->vector_name,
              temp_vreg,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset);
        } else {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
              LIBXSMM_X86_INSTR_VMULPS,
              i_micro_kernel_config->vector_name,
              i_gp_reg_mapping->gp_reg_scale_base,
              i_gp_reg_mapping->gp_reg_n_loop,
              4, 0, 1,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
              (apply_op == 1) ? ((apply_redop == 1) ? im + vecidxin_offset : im + vecin_offset) : im + vecidxin_offset,
              0, 0, 0);
        }
      }

      /* Now apply the Reduce OP */
      if (apply_redop == 1) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
            reduceop_instr,
            i_micro_kernel_config->vector_name,
            im + vecidxin_offset,
            im + vecout_offset,
            im + vecout_offset,
            reduceop_mask,
            reduceop_mask_cntl,
            reduceop_sae_cntl,
            reduceop_imm);
      }
    }

    libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);

    /* Now store accumulators  */
    for (im = 0; im < peeled_m_trips; im++) {
      char vname = (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ? 'y' : 'z';
      if (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) {
        if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CPX) {
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VCVTNEPS2BF16,
              i_micro_kernel_config->vector_name,
              im + vecout_offset, LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecout_offset,
              0);
        } else {
          libxsmm_generator_cvtfp32bf16_avx512_replacement_sequence( io_generated_code, i_micro_kernel_config, im + vecout_offset, 30, 31 );
          /* shift FP32 by 16bit to right */
          libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPSRAD_I,
              i_micro_kernel_config->vector_name,
              im + vecout_offset,
              LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecout_offset,
              16);
          /* store 16 bit values into lower portion of reg_0 */
          libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VPMOVDW,
              i_micro_kernel_config->vector_name,
              im + vecout_offset, LIBXSMM_X86_VEC_REG_UNDEF,
              im + vecout_offset,
              LIBXSMM_X86_VEC_REG_UNDEF);
        }
      }
      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          ((im == peeled_m_trips-1) && (use_m_masking == 1)) ? i_micro_kernel_config->vmove_instruction_out : vstore_instr,
          i_gp_reg_mapping->gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * 16  * i_micro_kernel_config->datatype_size_out,
          vname,
          im + vecout_offset, (im == peeled_m_trips-1) ? use_m_masking : 0, 0, 1 );
    }
  }

  if (io_generated_code->arch < LIBXSMM_X86_AVX512_CPX) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
  }

  if (pushpop_scaleop_reg == 1) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_mapping->gp_reg_scale_base );
  }

  libxsmm_x86_instruction_register_jump_label(io_generated_code, END_LABEL, &l_jump_label_tracker);
}

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_index_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {

  unsigned int m, im, use_m_masking, m_trips, max_m_unrolling = 4, m_unroll_factor = 1, m_trips_loop = 0, peeled_m_trips = 0, mask_out_count = 0;
  unsigned int idx_tsize =  i_mateltwise_desc->n;
  unsigned int ind_alu_mov_instr = (idx_tsize == 8) ? LIBXSMM_X86_INSTR_MOVQ : LIBXSMM_X86_INSTR_MOVL;
  int pf_dist = 4, use_nts = 0, pf_instr = LIBXSMM_X86_INSTR_PREFETCHT1, pf_type = 1, load_acc = 1;
  unsigned int vstore_instr = 0;
  unsigned int NO_PF_LABEL_START = 0;
  unsigned int NO_PF_LABEL_START_2 = 1;
  unsigned int END_LABEL = 2;
  const char *const env_max_m_unroll = getenv("MAX_M_UNROLL_REDUCE_COLS_IDX");
  const char *const env_pf_dist = getenv("PF_DIST_REDUCE_COLS_IDX");
  const char *const env_pf_type = getenv("PF_TYPE_REDUCE_COLS_IDX");
  const char *const env_nts     = getenv("NTS_REDUCE_COLS_IDX");
  const char *const env_load_acc= getenv("LOAD_ACCS_REDUCE_COLS_IDX");
  libxsmm_jump_label_tracker l_jump_label_tracker;
  libxsmm_reset_jump_label_tracker(&l_jump_label_tracker);

  if ( 0 == env_max_m_unroll ) {
  } else {
    max_m_unrolling = atoi(env_max_m_unroll);
  }
  if ( 0 == env_pf_dist ) {
  } else {
    pf_dist = atoi(env_pf_dist);
  }
  if ( 0 == env_pf_type ) {
  } else {
    pf_type = atoi(env_pf_type);
  }
  if ( 0 == env_nts ) {
  } else {
    use_nts = atoi(env_nts);
  }
  if ( 0 == env_load_acc ) {
  } else {
    load_acc = atoi(env_load_acc);
  }

  pf_instr      = (pf_type == 2) ? LIBXSMM_X86_INSTR_PREFETCHT1 : LIBXSMM_X86_INSTR_PREFETCHT0 ;
  vstore_instr  = (use_nts == 0) ? i_micro_kernel_config->vmove_instruction_out : LIBXSMM_X86_INSTR_VMOVNTPS;

  i_gp_reg_mapping->gp_reg_n        = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_ind_base = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_in_base  = LIBXSMM_X86_GP_REG_R10;
  i_gp_reg_mapping->gp_reg_out      = LIBXSMM_X86_GP_REG_R11;
  i_gp_reg_mapping->gp_reg_m_loop   = LIBXSMM_X86_GP_REG_RAX;
  i_gp_reg_mapping->gp_reg_n_loop   = LIBXSMM_X86_GP_REG_RDX;
  i_gp_reg_mapping->gp_reg_in       = LIBXSMM_X86_GP_REG_RSI;
  i_gp_reg_mapping->gp_reg_in_pf    = LIBXSMM_X86_GP_REG_RCX;

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_n,
      0 );

  libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_n, 0);
  libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JLE, END_LABEL, &l_jump_label_tracker);

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      8,
      i_gp_reg_mapping->gp_reg_ind_base,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      16,
      i_gp_reg_mapping->gp_reg_in_base,
      0 );

  libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      24,
      i_gp_reg_mapping->gp_reg_out,
      0 );

  m                 = i_mateltwise_desc->m;
  use_m_masking     = ( m % 16 == 0 ) ? 0 : 1;
  m_trips           = (m + 15) / 16;
  m_unroll_factor   = m_trips;
  m_trips_loop      = 1;
  peeled_m_trips    = 0;

  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for reading/output-writing */
    mask_out_count = 16 - (m % 16);
    libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_RAX, 1, mask_out_count, LIBXSMM_GEMM_PRECISION_F32);
  }

  /* In this case we have to generate a loop for m */
  if (m_unroll_factor > max_m_unrolling) {
    m_unroll_factor = max_m_unrolling;
    m_trips_loop = m_trips/m_unroll_factor;
    peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  } else {
    if ((use_m_masking > 0) && (peeled_m_trips == 0)) {
      m_trips_loop--;
      peeled_m_trips = m_trips  - m_unroll_factor * m_trips_loop;
    }
  }

  if (m_trips_loop > 1) {
    libxsmm_generator_mateltwise_header_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop );
  }

  if (m_trips_loop >= 1) {
    for (im = 0; im < m_unroll_factor; im++) {
      if (load_acc == 0) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPXORD,
                                               i_micro_kernel_config->vector_name,
                                               im, im , im);
      } else {
        if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                             LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                             'z',
                                                             i_gp_reg_mapping->gp_reg_out,
                                                             LIBXSMM_X86_GP_REG_UNDEF,
                                                             0,
                                                             im * 16  * i_micro_kernel_config->datatype_size_out,
                                                             0,
                                                             LIBXSMM_X86_VEC_REG_UNDEF,
                                                             im,
                                                             0,
                                                             0,
                                                             0);
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_out,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * 16  * i_micro_kernel_config->datatype_size_out,
              ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) ? 'y' :  i_micro_kernel_config->vector_name,
              im, 0, 0, 0 );
        }
      }
    }

    if (pf_dist > 0) {
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_n_loop, 0);
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
      libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JLE, NO_PF_LABEL_START, &l_jump_label_tracker);

      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mapping->gp_reg_n, pf_dist);
      libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, 1);
      libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, i_gp_reg_mapping->gp_reg_in, 0);
      libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, pf_dist * idx_tsize, i_gp_reg_mapping->gp_reg_in_pf, 0);
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in_pf, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in);
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in_pf);

      for (im = 0; im < m_unroll_factor; im++) {
        if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                             LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                             'z',
                                                             i_gp_reg_mapping->gp_reg_in,
                                                             LIBXSMM_X86_GP_REG_UNDEF,
                                                             0,
                                                             im * 16  * i_micro_kernel_config->datatype_size_in,
                                                             0,
                                                             LIBXSMM_X86_VEC_REG_UNDEF,
                                                             im+16,
                                                             0,
                                                             0,
                                                             0);

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        LIBXSMM_X86_INSTR_VADDPS,
                                        i_micro_kernel_config->vector_name,
                                        im, 16+im, im );
        } else {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VADDPS,
            0,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            im,
            im );
        }

        if ((im * 16 * i_micro_kernel_config->datatype_size_in) % 64 == 0 ) {
          libxsmm_x86_instruction_prefetch(io_generated_code,
              pf_instr,
              i_gp_reg_mapping->gp_reg_in_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * 16 * i_micro_kernel_config->datatype_size_in);
        }
      }

      libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
      /* NO_PF_LABEL_START */
      libxsmm_x86_instruction_register_jump_label(io_generated_code, NO_PF_LABEL_START, &l_jump_label_tracker);
    }

    /* Perform the reductions for all columns */
    libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, (pf_dist > 0) ? 1 : 0);
    libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, i_gp_reg_mapping->gp_reg_in, 0);
    libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in);
    libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in);

    for (im = 0; im < m_unroll_factor; im++) {
      if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
        libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                           LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                           'z',
                                                           i_gp_reg_mapping->gp_reg_in,
                                                           LIBXSMM_X86_GP_REG_UNDEF,
                                                           0,
                                                           im * 16 * i_micro_kernel_config->datatype_size_in,
                                                           0,
                                                           LIBXSMM_X86_VEC_REG_UNDEF,
                                                           16+im,
                                                           0,
                                                           0,
                                                           0);

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                      i_micro_kernel_config->instruction_set,
                                      LIBXSMM_X86_INSTR_VADDPS,
                                      i_micro_kernel_config->vector_name,
                                      im, 16+im, im );
      } else {
        libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
          i_micro_kernel_config->instruction_set,
          LIBXSMM_X86_INSTR_VADDPS,
          0,
          i_gp_reg_mapping->gp_reg_in,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          im * 16 * i_micro_kernel_config->datatype_size_in,
          i_micro_kernel_config->vector_name,
          im,
          im );
      }
    }

    libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);

    /* Now store accumulators  */
    for (im = 0; im < m_unroll_factor; im++) {
      if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
        libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
                                                             LIBXSMM_X86_INSTR_VCVTPS2PH,
                                                             i_micro_kernel_config->vector_name,
                                                             i_gp_reg_mapping->gp_reg_out,
                                                             LIBXSMM_X86_GP_REG_UNDEF,
                                                             0,
                                                             im * 16  * i_micro_kernel_config->datatype_size_out,
                                                             0,
                                                             LIBXSMM_X86_VEC_REG_UNDEF,
                                                             im,
                                                             0,
                                                             0,
                                                             3 );
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            vstore_instr,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16  * i_micro_kernel_config->datatype_size_out,
            ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) ? 'y' :  i_micro_kernel_config->vector_name,
            im, 0, 0, 1 );
      }
    }
  }

  if (m_trips_loop > 1) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, m_unroll_factor * 16 * i_micro_kernel_config->datatype_size_out);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, m_unroll_factor * 16 * i_micro_kernel_config->datatype_size_in);
    libxsmm_generator_mateltwise_footer_m_loop( io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_m_loop, m_trips_loop);
  }

  if (peeled_m_trips > 0) {
    if (m_trips_loop == 1) {
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_out, m_unroll_factor * 16 * i_micro_kernel_config->datatype_size_out);
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, m_unroll_factor * 16 * i_micro_kernel_config->datatype_size_in);
    }

    /* Perform the reductions for all columns */
    for (im = 0; im < peeled_m_trips; im++) {
      if (load_acc == 0) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                               i_micro_kernel_config->instruction_set,
                                               LIBXSMM_X86_INSTR_VPXORD,
                                               i_micro_kernel_config->vector_name,
                                               im, im , im);
      } else {
        if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                             LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                             'z',
                                                             i_gp_reg_mapping->gp_reg_out,
                                                             LIBXSMM_X86_GP_REG_UNDEF,
                                                             0,
                                                             im * 16  * i_micro_kernel_config->datatype_size_out,
                                                             0,
                                                             LIBXSMM_X86_VEC_REG_UNDEF,
                                                             im,
                                                             (im == peeled_m_trips-1) ? use_m_masking : 0,
                                                             0,
                                                             0);
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_out,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * 16  * i_micro_kernel_config->datatype_size_out,
              ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) ? 'y' :  i_micro_kernel_config->vector_name,
              im, (im == peeled_m_trips-1) ? use_m_masking : 0, 0, 0 );
        }
      }
    }

    if (pf_dist > 0) {
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_n_loop, 0);
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
      libxsmm_x86_instruction_jump_to_label(io_generated_code, LIBXSMM_X86_INSTR_JLE, NO_PF_LABEL_START_2, &l_jump_label_tracker);

      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_SUBQ, i_gp_reg_mapping->gp_reg_n, pf_dist);
      libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, 1);
      libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, i_gp_reg_mapping->gp_reg_in, 0);
      libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, pf_dist * idx_tsize, i_gp_reg_mapping->gp_reg_in_pf, 0);
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in_pf, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in);
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in);
      libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in_pf);

      for (im = 0; im < peeled_m_trips; im++) {
        if ((im == peeled_m_trips -1) && (use_m_masking > 0)) {
          if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                           LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                           'z',
                                                           i_gp_reg_mapping->gp_reg_in,
                                                           LIBXSMM_X86_GP_REG_UNDEF,
                                                           0,
                                                           im * 16 * i_micro_kernel_config->datatype_size_in,
                                                           0,
                                                           LIBXSMM_X86_VEC_REG_UNDEF,
                                                           16+im,
                                                           use_m_masking,
                                                           0,
                                                           0);
          } else {
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                i_micro_kernel_config->vmove_instruction_in,
                i_gp_reg_mapping->gp_reg_in,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                im * 16 * i_micro_kernel_config->datatype_size_in,
                ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) ? 'y' :  i_micro_kernel_config->vector_name,
                im+16, use_m_masking, 1, 0 );

          }

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          LIBXSMM_X86_INSTR_VADDPS,
                                          i_micro_kernel_config->vector_name,
                                          im, im+16, im );
        } else {
          if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                           LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                           'z',
                                                           i_gp_reg_mapping->gp_reg_in,
                                                           LIBXSMM_X86_GP_REG_UNDEF,
                                                           0,
                                                           im * 16 * i_micro_kernel_config->datatype_size_in,
                                                           0,
                                                           LIBXSMM_X86_VEC_REG_UNDEF,
                                                           16+im,
                                                           0,
                                                           0,
                                                           0);

            libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                          i_micro_kernel_config->instruction_set,
                                          LIBXSMM_X86_INSTR_VADDPS,
                                          i_micro_kernel_config->vector_name,
                                          im, 16+im, im );
          } else {
            libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
              i_micro_kernel_config->instruction_set,
              LIBXSMM_X86_INSTR_VADDPS,
              0,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * 16 * i_micro_kernel_config->datatype_size_in,
              i_micro_kernel_config->vector_name,
              im,
              im );
          }
        }

        if ((im * 16 * i_micro_kernel_config->datatype_size_in) % 64 == 0 ) {
          libxsmm_x86_instruction_prefetch(io_generated_code,
              pf_instr,
              i_gp_reg_mapping->gp_reg_in_pf,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * 16 * i_micro_kernel_config->datatype_size_in);
        }
      }

      libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);
      libxsmm_x86_instruction_alu_imm(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_n, pf_dist);
      /* NO_PF_LABEL_START_2 */
      libxsmm_x86_instruction_register_jump_label(io_generated_code, NO_PF_LABEL_START_2, &l_jump_label_tracker);
    }

    libxsmm_generator_mateltwise_header_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, (pf_dist > 0) ? 1 : 0);
    libxsmm_x86_instruction_alu_mem(io_generated_code, ind_alu_mov_instr, i_gp_reg_mapping->gp_reg_ind_base, i_gp_reg_mapping->gp_reg_n_loop, idx_tsize, 0, i_gp_reg_mapping->gp_reg_in, 0);
    libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_IMUL, i_gp_reg_mapping->gp_reg_in, i_mateltwise_desc->ldi * i_micro_kernel_config->datatype_size_in);
    libxsmm_x86_instruction_alu_reg(io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_in_base, i_gp_reg_mapping->gp_reg_in);

    for (im = 0; im < peeled_m_trips; im++) {
      if ((im == peeled_m_trips -1) && (use_m_masking > 0)) {
        if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                         LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                         'z',
                                                         i_gp_reg_mapping->gp_reg_in,
                                                         LIBXSMM_X86_GP_REG_UNDEF,
                                                         0,
                                                         im * 16 * i_micro_kernel_config->datatype_size_in,
                                                         0,
                                                         LIBXSMM_X86_VEC_REG_UNDEF,
                                                         16+im,
                                                         use_m_masking,
                                                         0,
                                                         0);
        } else {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              im * 16 * i_micro_kernel_config->datatype_size_in,
              ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) ? 'y' :  i_micro_kernel_config->vector_name,
              16+im, use_m_masking, 1, 0 );
        }

        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        LIBXSMM_X86_INSTR_VADDPS,
                                        i_micro_kernel_config->vector_name,
                                        im, 16+im, im );
      } else {
        if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8(io_generated_code,
                                                         LIBXSMM_X86_INSTR_VCVTPH2PS,
                                                         'z',
                                                         i_gp_reg_mapping->gp_reg_in,
                                                         LIBXSMM_X86_GP_REG_UNDEF,
                                                         0,
                                                         im * 16 * i_micro_kernel_config->datatype_size_in,
                                                         0,
                                                         LIBXSMM_X86_VEC_REG_UNDEF,
                                                         16+im,
                                                         0,
                                                         0,
                                                         0);

          libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                        i_micro_kernel_config->instruction_set,
                                        LIBXSMM_X86_INSTR_VADDPS,
                                        i_micro_kernel_config->vector_name,
                                        im, 16+im, im );
        } else {
          libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VADDPS,
            0,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16 * i_micro_kernel_config->datatype_size_in,
            i_micro_kernel_config->vector_name,
            im,
            im );
        }
      }
    }

    libxsmm_generator_mateltwise_footer_n_dyn_loop(io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, i_gp_reg_mapping->gp_reg_n);

    /* Now store accumulators  */
    for (im = 0; im < peeled_m_trips; im++) {
      if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
        libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code,
                                                             LIBXSMM_X86_INSTR_VCVTPS2PH,
                                                             i_micro_kernel_config->vector_name,
                                                             i_gp_reg_mapping->gp_reg_out,
                                                             LIBXSMM_X86_GP_REG_UNDEF,
                                                             0,
                                                             im * 16  * i_micro_kernel_config->datatype_size_out,
                                                             0,
                                                             LIBXSMM_X86_VEC_REG_UNDEF,
                                                             im,
                                                             (im == peeled_m_trips-1) ? use_m_masking : 0,
                                                             0,
                                                             3 );
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            ((im == peeled_m_trips-1) && (use_m_masking ==1)) ? i_micro_kernel_config->vmove_instruction_out : vstore_instr,
            i_gp_reg_mapping->gp_reg_out,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * 16  * i_micro_kernel_config->datatype_size_out,
            ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) ? 'y' :  i_micro_kernel_config->vector_name,
            im, (im == peeled_m_trips-1) ? use_m_masking : 0, 0, 1 );
      }
    }
  }

  libxsmm_x86_instruction_register_jump_label(io_generated_code, END_LABEL, &l_jump_label_tracker);
}

LIBXSMM_API_INTERN
void libxsmm_generator_copy_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  unsigned int m, n, im, in, in_load, unroll_iter, use_m_masking = 0, m_trips = 0, n_trips = 0, vlen_in = 32, vlen_out = 32, mask_in_count = 0, mask_out_count = 0;
  unsigned int reserved_zmms = 0, max_nm_unrolling = 32, n_unroll_factor = 1;
  unsigned int mask_reg_in = 1, mask_reg_out = 2;
  unsigned int zero_vreg = 0, vreg_in = 0;
  unsigned int read_input = 1;
  unsigned int colbcast = 0;
  unsigned int hoist_colbcast = 0;
  char     input_vname = 'z', output_vname = 'z';

  /* Some rudimentary checking of M, N and LDs*/
  if ( ((i_mateltwise_desc->m > i_mateltwise_desc->ldi) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_COPY_ZERO) == 0) ) || (i_mateltwise_desc->m > i_mateltwise_desc->ldo) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
    return;
  }

  /* Specify vlens depending on precision  */
  if ( (LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype))) {
    vlen_in = 8;
  } else if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype))) {
    vlen_in = 16;
  } else if ( (LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype))) {
    vlen_in = 32;
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) {
    vlen_in = 64;
  }

  if ( (LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_I64 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype))) {
    vlen_out = 8;
  } else if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype))) {
    vlen_out = 16;
  } else if ( (LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) || (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype))) {
    vlen_out = 32;
  } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) {
    vlen_out = 64;
  }

  if (vlen_in != vlen_out) {
    if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) ) {
      vlen_in = 16;
      input_vname = 'y';
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  }

  /* Check if we have to hoist column broadcast */
  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_COPY_COLBCAST) > 0) {
    colbcast = 1;
  }

  /* Configure the register mapping for this eltwise kernel */
  i_gp_reg_mapping->gp_reg_in     = LIBXSMM_X86_GP_REG_R8;
  i_gp_reg_mapping->gp_reg_out    = LIBXSMM_X86_GP_REG_R9;
  i_gp_reg_mapping->gp_reg_n_loop = LIBXSMM_X86_GP_REG_R10;

  /* Load the input pointer and output pointer */
  /* Load input only if we dont have zerobeta  */
  if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_COPY_ZERO) > 0) {
    zero_vreg = reserved_zmms++;
    reserved_zmms++;
    read_input = 0;
    libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                             i_micro_kernel_config->instruction_set,
                                             LIBXSMM_X86_INSTR_VPXORD,
                                             i_micro_kernel_config->vector_name,
                                             zero_vreg, zero_vreg, zero_vreg );
  } else {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
      i_micro_kernel_config->alu_mov_instruction,
      i_gp_reg_mapping->gp_reg_param_struct,
      LIBXSMM_X86_GP_REG_UNDEF, 0,
      0,
      i_gp_reg_mapping->gp_reg_in,
      0 );
  }

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
  use_m_masking     = ( m % vlen_in == 0 ) ? 0 : 1;
  m_trips           = (m + vlen_in - 1) / vlen_in;

  max_nm_unrolling = max_nm_unrolling - reserved_zmms;

  if (m_trips > max_nm_unrolling) {
    n_unroll_factor = 1;
  } else {
    /* Explore n unrolling opportunities... We unroll only by factors that divide N  */
    n_unroll_factor = n;
    while (m_trips * n_unroll_factor > max_nm_unrolling) {
      n_unroll_factor--;
    }
    while (n % n_unroll_factor > 0) {
      n_unroll_factor--;
    }
  }
  n_trips = n / n_unroll_factor;

  /* Calculate input and output masks in case we see m_masking */
  if (use_m_masking == 1) {
    /* Calculate mask reg 1 for input-reading */
    if (read_input > 0) {
      mask_in_count = vlen_in - m % vlen_in;
      libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R11, mask_reg_in, mask_in_count, LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype));
    }
    /* Calculate mask reg 2 for output-writing */
    if (((read_input > 0) && (vlen_in != vlen_out)) || (read_input == 0)){
      mask_out_count = vlen_out - m % vlen_out;
      libxsmm_generator_mateltwise_initialize_avx512_mask(io_generated_code, LIBXSMM_X86_GP_REG_R11, mask_reg_out, mask_out_count, LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype));
    } else {
      mask_reg_out = mask_reg_in;
    }
  }

  /* Check if we can hoist column brodcast to avoid redundant loads  */
  if ((colbcast == 1) && (m_trips <= max_nm_unrolling) && (n_trips >= 1)) {
    hoist_colbcast = 1;
  }

  if (hoist_colbcast == 1) {
    for (im = 0; im < m_trips; im++) {
      unroll_iter = im;
      vreg_in = unroll_iter % (max_nm_unrolling - reserved_zmms) + reserved_zmms;
      if (read_input == 0) {
        vreg_in = zero_vreg;
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
            i_micro_kernel_config->instruction_set,
            i_micro_kernel_config->vmove_instruction_in,
            i_gp_reg_mapping->gp_reg_in,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            im * vlen_in * i_micro_kernel_config->datatype_size_in,
            input_vname,
            vreg_in, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_in : 0, 1, 0 );
      }

      if ((read_input > 0) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) ) {
        /* convert 16 bit values into 32 bit (integer convert) */
        libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPMOVSXWD,
            i_micro_kernel_config->vector_name,
            vreg_in, LIBXSMM_X86_VEC_REG_UNDEF,
            vreg_in,
            LIBXSMM_X86_VEC_REG_UNDEF);

        /* shift 16 bits to the left to generate valid FP32 numbers */
        libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPSLLD_I,
            i_micro_kernel_config->vector_name,
            vreg_in,
            LIBXSMM_X86_VEC_REG_UNDEF,
            vreg_in,
            16);
      }
    }
  }

  if (n_trips > 1) {
    /* open n loop */
    libxsmm_generator_mateltwise_header_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop );
  }

  for (in = 0; in < n_unroll_factor; in++) {
    for (im = 0; im < m_trips; im++) {
      in_load = (colbcast == 0) ? in : 0;
      unroll_iter = in_load * m_trips + im;
      vreg_in = unroll_iter % (max_nm_unrolling - reserved_zmms) + reserved_zmms;
      if (read_input == 0) {
        vreg_in = zero_vreg;
      } else {
        if (hoist_colbcast == 0) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
              i_micro_kernel_config->instruction_set,
              i_micro_kernel_config->vmove_instruction_in,
              i_gp_reg_mapping->gp_reg_in,
              LIBXSMM_X86_GP_REG_UNDEF, 0,
              (im * vlen_in + in_load * i_mateltwise_desc->ldi) * i_micro_kernel_config->datatype_size_in,
              input_vname,
              vreg_in, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_in : 0, 1, 0 );
        }
      }

      if ((read_input > 0) && (hoist_colbcast == 0) && (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(i_mateltwise_desc->datatype)) && (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP(i_mateltwise_desc->datatype)) ) {
        /* convert 16 bit values into 32 bit (integer convert) */
        libxsmm_x86_instruction_vec_compute_convert( io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPMOVSXWD,
            i_micro_kernel_config->vector_name,
            vreg_in, LIBXSMM_X86_VEC_REG_UNDEF,
            vreg_in,
            LIBXSMM_X86_VEC_REG_UNDEF);

        /* shift 16 bits to the left to generate valid FP32 numbers */
        libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code,
            i_micro_kernel_config->instruction_set,
            LIBXSMM_X86_INSTR_VPSLLD_I,
            i_micro_kernel_config->vector_name,
            vreg_in,
            LIBXSMM_X86_VEC_REG_UNDEF,
            vreg_in,
            16);
      }

      libxsmm_x86_instruction_vec_move( io_generated_code,
          i_micro_kernel_config->instruction_set,
          i_micro_kernel_config->vmove_instruction_out,
          i_gp_reg_mapping->gp_reg_out,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          (im * vlen_out + in * i_mateltwise_desc->ldo) * i_micro_kernel_config->datatype_size_out,
          output_vname,
          vreg_in, ((im == (m_trips-1)) && (use_m_masking == 1)) ? mask_reg_out : 0, 0, 1 );

    }
  }

  if (n_trips > 1) {
    /* Adjust input and output pointer */
    if ((read_input > 0) && (colbcast == 0)) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          i_micro_kernel_config->alu_add_instruction,
          i_gp_reg_mapping->gp_reg_in,
          i_mateltwise_desc->ldi * n_unroll_factor * i_micro_kernel_config->datatype_size_in);
    }

    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        i_micro_kernel_config->alu_add_instruction,
        i_gp_reg_mapping->gp_reg_out,
        i_mateltwise_desc->ldo *  n_unroll_factor * i_micro_kernel_config->datatype_size_out);

    /* close n loop */
    libxsmm_generator_mateltwise_footer_n_loop(  io_generated_code, io_loop_label_tracker, i_micro_kernel_config, i_gp_reg_mapping->gp_reg_n_loop, n_trips);
  }
}


LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_avx_avx512_kernel( libxsmm_generated_code*         io_generated_code,
    const libxsmm_meltw_descriptor* i_mateltwise_desc) {
  libxsmm_mateltwise_kernel_config  l_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker        l_loop_label_tracker;
  unsigned int skip_pushpops = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE_COLS_IDX) ||
                                (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX) ||
                                (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_COPY) ||
                                (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE) ||
                                (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TRANSFORM) ||
                                ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_SCALE) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_ROWS_BCASTVAL_ACCUMULATE) > 0)) ||
                                ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_CVT_VNNI_FORMAT) > 0) )) ? 1 : 0;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif

  /* for now we only support AVX512_CORE and higher */
  if ( io_generated_code->arch < LIBXSMM_X86_AVX512_CORE ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* define mateltwise kernel config */
  libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_kernel_config, io_generated_code->arch, i_mateltwise_desc);

  /* open asm */
  libxsmm_x86_instruction_open_stream_mateltwise( io_generated_code, l_gp_reg_mapping.gp_reg_param_struct, NULL, skip_pushpops );

  /* Depending on the elementwise function, dispatch the proper code JITer */
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16) || (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16_ACT) || (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_ACT_CVTFP32BF16)) {
    if ( (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_CVT_VNNI_FORMAT) > 0) ) {
        libxsmm_generator_cvtfp32bf16_vnni_format_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else {
        libxsmm_generator_cvtfp32bf16_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      }
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE) {
    if ( (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ROWS) > 0) {
        libxsmm_generator_reduce_rows_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_COLS) > 0) {
        libxsmm_generator_reduce_cols_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else {
        /* This should not happen  */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
        return;
      }
    } else {
      if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype))) &&
           ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_COLS) > 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ELTS) > 0) &&
           ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ROWS) == 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ELTS_SQUARED) == 0) &&
           ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_OP_ADD) > 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_NCNC_FORMAT) > 0) ) {
        libxsmm_generator_reduce_cols_ncnc_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else {
        /* This should not happen  */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_COPY) {
    libxsmm_generator_copy_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE_COLS_IDX) {
    libxsmm_generator_reduce_cols_index_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX) {
    libxsmm_generator_opreduce_vecs_index_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_SCALE) {
    if ( (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      libxsmm_generator_scale_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_RELU) {
    if ( (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      libxsmm_generator_relu_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TRANSFORM ) {
    libxsmm_generator_transform_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_DROPOUT ) {
    libxsmm_generator_dropout_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else  {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_mateltwise( io_generated_code, NULL, skip_pushpops);
}

