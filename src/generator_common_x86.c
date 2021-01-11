/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include "generator_common_x86.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_haddps_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_inout,
    const unsigned int                             i_vec_tmp1,
    const unsigned int                             i_vec_tmp2) {

  if (i_vec_tmp1 > 15 || i_vec_tmp2 > 15 ) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           'z',
                                           i_vec_inout, i_vec_inout, i_vec_tmp1, 0x4e );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           'z',
                                           i_vec_inout, i_vec_tmp1, i_vec_inout );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           'z',
                                           i_vec_inout, i_vec_inout, i_vec_tmp1, 0xb1 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           'z',
                                           i_vec_inout, i_vec_tmp1, i_vec_tmp2 );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp2, i_vec_tmp1, 0x4e );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp1, i_vec_tmp2 );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp2, i_vec_tmp1, 0x1 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           LIBXSMM_X86_INSTR_VADDPS,
                                           'z',
                                           i_vec_tmp2, i_vec_tmp1, i_vec_inout );
}

LIBXSMM_API_INTERN
void libxsmm_generator_generic_loop_header( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const unsigned int                 i_loop_reg,
    const unsigned int                 i_loop_init_val,
    const unsigned int                 i_loop_step ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_loop_reg, i_loop_init_val );
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_loop_reg, i_loop_step);
}

LIBXSMM_API_INTERN
void libxsmm_generator_generic_loop_footer( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const unsigned int                 i_loop_reg,
    const unsigned int                 i_loop_bound) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_CMPQ, i_loop_reg, i_loop_bound );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, LIBXSMM_X86_INSTR_JL, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128pp_avx512( libxsmm_generated_code* io_generated_code,
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
void libxsmm_generator_xoshiro128p_f32_avx512( libxsmm_generated_code* io_generated_code,
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
void libxsmm_generator_cvtbf16ps_avx512( libxsmm_generated_code* io_generated_code,
                                         const char              i_vname,
                                         const unsigned int      i_vec_reg,
                                         const unsigned int      o_vec_reg ) {
  /* @TODO check for valid i_vnames */

  /* convert 16 bit values into 32 bit (integer convert) */
  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXWD, i_vname,
                                            i_vec_reg, o_vec_reg );

  /* shift 16 bits to the left to generate valid FP32 numbers */
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_vname,
                                                 o_vec_reg, o_vec_reg, 16 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( libxsmm_generated_code* io_generated_code,
                                                        const unsigned int      io_gp_reg ) {
  /* init stack with helper variables for SW-based RNE rounding */
  /* push 0x7f800000 on the stack, naninf masking */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, io_gp_reg, 0x7f800000);
  libxsmm_x86_instruction_push_reg( io_generated_code, io_gp_reg );

  /* push 0x00010000 on the stack, fixup masking */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, io_gp_reg, 0x00010000);
  libxsmm_x86_instruction_push_reg( io_generated_code, io_gp_reg );

  /* push 0x00007fff on the stack, rneadd */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, io_gp_reg, 0x00007fff);
  libxsmm_x86_instruction_push_reg( io_generated_code, io_gp_reg);

  /* push 0x00000001 on the stack, fixup */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, io_gp_reg, 0x00000001);
  libxsmm_x86_instruction_push_reg( io_generated_code, io_gp_reg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( libxsmm_generated_code* io_generated_code,
                                                         const unsigned int      io_gp_reg ) {
  libxsmm_x86_instruction_pop_reg( io_generated_code, io_gp_reg );
  libxsmm_x86_instruction_pop_reg( io_generated_code, io_gp_reg );
  libxsmm_x86_instruction_pop_reg( io_generated_code, io_gp_reg );
  libxsmm_x86_instruction_pop_reg( io_generated_code, io_gp_reg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( libxsmm_generated_code* io_generated_code,
                                                          const char              i_vname,
                                                          const unsigned int      i_vec_reg,
                                                          const unsigned int      o_vec_reg,
                                                          const unsigned int      io_vec_tmp_0,
                                                          const unsigned int      io_vec_tmp_1,
                                                          const unsigned int      io_mask_0,
                                                          const unsigned int      io_mask_1 ) {
  /* @TODO check for valid i_vnames */

  /* and with naninf */
  libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPANDD, i_vname,
                                                LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 24, 1,
                                                i_vec_reg, io_vec_tmp_0 );

  /* and with fixup */
  libxsmm_x86_instruction_vec_compute_mem_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPANDD, i_vname,
                                                LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 16, 1,
                                                i_vec_reg, io_vec_tmp_1 );

  /* compute naninf mask */
  libxsmm_x86_instruction_vec_compute_mem_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPCMPD, i_vname,
                                                     LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 24, 1,
                                                     io_vec_tmp_0, io_mask_0, 4 );

  /* compute fixup mask */
  libxsmm_x86_instruction_vec_compute_mem_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPCMPD, i_vname,
                                                     LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 16, 1,
                                                     io_vec_tmp_1, io_mask_1, 0 );

  /* load rneadd */
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, i_vname,
                                    io_vec_tmp_0, 0, 1, 0 );

  /* load fixup */
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, i_vname,
                                    io_vec_tmp_1, 0, 1, 0 );

  /* compute fixup */
  libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, LIBXSMM_X86_INSTR_VPADDD, i_vname,
                                                 io_vec_tmp_1, io_vec_tmp_0, io_vec_tmp_0, io_mask_1, 0 );

  if ( i_vec_reg != o_vec_reg ) {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64_LD, i_vname,
                                              i_vec_reg, o_vec_reg );
  }

  /* compute fixup */
  libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code, LIBXSMM_X86_INSTR_VPADDD, i_vname,
                                                 io_vec_tmp_0, i_vec_reg, o_vec_reg, io_mask_0, 0 );

  /* shift FP32 by 16bit to right */
  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRAD_I, i_vname,
                                                 o_vec_reg, o_vec_reg, 16 );

  /* store 16 bit values into lower portion of reg_0 */
  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VPMOVDW, i_vname,
                                            o_vec_reg, o_vec_reg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtneps2bf16_avx512( libxsmm_generated_code* io_generated_code,
                                             const char              i_vname,
                                             const unsigned int      i_vec_reg,
                                             const unsigned int      o_vec_teg,
                                             const unsigned int      io_gp_reg,
                                             const unsigned int      io_vec_tmp_0,
                                             const unsigned int      io_vec_tmp_1,
                                             const unsigned int      io_mask_0,
                                             const unsigned int      io_mask_1 ) {
  libxsmm_generator_vcvtneps2bf16_avx512_prep_stack( io_generated_code, io_gp_reg );

  libxsmm_generator_vcvtneps2bf16_avx512_preppedstack( io_generated_code, i_vname, i_vec_reg, o_vec_teg,
                                                       io_vec_tmp_0, io_vec_tmp_1, io_mask_0, io_mask_1 );

  libxsmm_generator_vcvtneps2bf16_avx512_clean_stack( io_generated_code, io_gp_reg );
}

LIBXSMM_API_INTERN
int libxsmm_generator_meltw_get_rbp_relative_offset( libxsmm_meltw_stack_var stack_var ) {
  /* The stack at exit of setup looks like this:
   *
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      Input ptr                                 <-- RBP-8
   *      Output ptr                                <-- RBP-16
   *      Mask ptr                                  <-- RBP-24
   *      Scratch ptr in stack (to be filled)       <-- RBP-32
   *
   * */

  switch ( stack_var ) {
    case LIBXSMM_MELTW_STACK_VAR_INP_PTR:
      return -8;
    case LIBXSMM_MELTW_STACK_VAR_OUT_PTR:
      return -16;
    case LIBXSMM_MELTW_STACK_VAR_MASK_PTR:
      return -24;
    case LIBXSMM_MELTW_STACK_VAR_SCRATCH_PTR:
      return -32;
    default:
      return 0;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_getval_stack_var( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meltw_stack_var            stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_meltw_get_rbp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_setval_stack_var( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meltw_stack_var            stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_meltw_get_rbp_relative_offset(stack_var);
  /* make sure we requested to set  a legal stack var */
  if (offset >= 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 1 );
}



