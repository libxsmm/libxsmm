/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include "generator_common_x86.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_x86_save_gpr_regs(libxsmm_generated_code*   io_generated_code,
    const unsigned short    i_save_bitmask) {
  if ( ( i_save_bitmask & 0x1 ) == 0x1 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  }
  if ( ( i_save_bitmask & 0x2 ) == 0x2 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RCX );
  }
  if ( ( i_save_bitmask & 0x4 ) == 0x4 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
  }
  if ( ( i_save_bitmask & 0x8 ) == 0x8 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBX );
  }
  if ( ( i_save_bitmask & 0x10 ) == 0x10 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSP );
  }
  if ( ( i_save_bitmask & 0x20 ) == 0x20 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  }
  if ( ( i_save_bitmask & 0x40 ) == 0x40 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
  }
  if ( ( i_save_bitmask & 0x80 ) == 0x80 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDI );
  }
  if ( ( i_save_bitmask & 0x100 ) == 0x100 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R8 );
  }
  if ( ( i_save_bitmask & 0x200 ) == 0x200 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R9 );
  }
  if ( ( i_save_bitmask & 0x400 ) == 0x400 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R10 );
  }
  if ( ( i_save_bitmask & 0x800 ) == 0x800 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R11 );
  }
  if ( ( i_save_bitmask & 0x1000 ) == 0x1000 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
  }
  if ( ( i_save_bitmask & 0x2000 ) == 0x2000 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
  }
  if ( ( i_save_bitmask & 0x4000 ) == 0x4000 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
  }
  if ( ( i_save_bitmask & 0x8000 ) == 0x8000 ) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_x86_restore_gpr_regs(libxsmm_generated_code*   io_generated_code,
    const unsigned short    i_restore_bitmask) {

  if ( ( i_restore_bitmask & 0x8000 ) == 0x8000 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
  }
  if ( ( i_restore_bitmask & 0x4000 ) == 0x4000 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
  }
  if ( ( i_restore_bitmask & 0x2000 ) == 0x2000 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
  }
  if ( ( i_restore_bitmask & 0x1000 ) == 0x1000 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
  }
  if ( ( i_restore_bitmask & 0x800 ) == 0x800 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R11 );
  }
  if ( ( i_restore_bitmask & 0x400 ) == 0x400 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R10 );
  }
  if ( ( i_restore_bitmask & 0x200 ) == 0x200 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R9 );
  }
  if ( ( i_restore_bitmask & 0x100 ) == 0x100 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R8 );
  }
  if ( ( i_restore_bitmask & 0x80 ) == 0x80 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDI );
  }
  if ( ( i_restore_bitmask & 0x40 ) == 0x40 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
  }
  if ( ( i_restore_bitmask & 0x20 ) == 0x20 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  }
  if ( ( i_restore_bitmask & 0x10 ) == 0x10 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSP );
  }
  if ( ( i_restore_bitmask & 0x8 ) == 0x8 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBX );
  }
  if ( ( i_restore_bitmask & 0x4 ) == 0x4 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDX );
  }
  if ( ( i_restore_bitmask & 0x2 ) == 0x2 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RCX );
  }
  if ( ( i_restore_bitmask & 0x1 ) == 0x1 ) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RAX );
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_unified_vec_move( libxsmm_generated_code* io_generated_code,
                                                const unsigned int      i_vmove_instr,
                                                const unsigned int      i_gp_reg_base,
                                                const unsigned int      i_reg_idx,
                                                const unsigned int      i_scale,
                                                const int               i_displacement,
                                                const char              i_vector_name,
                                                const unsigned int      i_vec_reg_number_0,
                                                const unsigned int      i_use_masking,
                                                const unsigned int      i_mask_reg_number,
                                                const unsigned int      i_is_store ) {
  unsigned int vmove_instr = i_vmove_instr;
  if (io_generated_code->arch <= LIBXSMM_X86_AVX2){
    if (i_use_masking > 0) {
      if (i_is_store > 0 ) {
        vmove_instr = LIBXSMM_X86_INSTR_VMASKMOVPS_ST;
        if (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVUPD) {
          vmove_instr = LIBXSMM_X86_INSTR_VMASKMOVPD_ST;
        }
      } else {
        vmove_instr = LIBXSMM_X86_INSTR_VMASKMOVPS;
        if (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVUPD) {
          vmove_instr = LIBXSMM_X86_INSTR_VMASKMOVPD;
        }
      }
    }
  }

  libxsmm_x86_instruction_vex_evex_mask_mov( io_generated_code, vmove_instr, i_gp_reg_base, i_reg_idx, i_scale, i_displacement, i_vector_name, i_vec_reg_number_0, i_use_masking, i_mask_reg_number, i_is_store );
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_thres,
    const unsigned int                             i_vec_absmask,
    const unsigned int                             i_vec_scale,
    const unsigned int                             i_vec_shifter,
    const unsigned int                             i_vec_half,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2 ) {
  unsigned int thres_array[16] = { 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff, 0x40879fff };
  unsigned int absmask_array[16] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
  unsigned int scale_array[16] = { 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1, 0x406a0ea1};
  unsigned int shifter_array[16] = { 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000 };
  unsigned int half_array[16] = { 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000};
  unsigned int c0_array[16] = { 0x3ecc4231u, 0x3ecc541cu, 0x3ecd6c48u, 0x3ed174c3u, 0x3ed9bd5du, 0x3ee5acd5u, 0x3ef2aeddu, 0x3efd5384u, 0x3f016724u, 0x3f00f778u, 0x3efb389eu, 0x3ef0464du, 0x3ee3014fu, 0x3ed50a78u, 0x3ec779dbu, 0x3ebae363u };
  unsigned int c1_array[16] = { 0xb7c7fb58u, 0xbacb9740u, 0xbc3e4b3au, 0xbd0d292au, 0xbd8bc5d0u, 0xbdd9978fu, 0xbe0f92d3u, 0xbe27b66du, 0xbe328ce7u, 0xbe3125bfu, 0xbe26dc9du, 0xbe17a056u, 0xbe06bdebu, 0xbdecc593u, 0xbdcf57aau, 0xbdb5ea3au };
  unsigned int c2_array[16] = { 0xbd877b85u, 0xbd7d9780u, 0xbd4cb70eu, 0xbd08a1e9u, 0xbc808857u, 0xb9476fd2u, 0x3c36f765u, 0x3c924160u, 0x3ca7b1fcu, 0x3ca5732cu, 0x3c95af63u, 0x3c8079f7u, 0x3c55fa4fu, 0x3c2fa86bu, 0x3c0fbb00u, 0x3bec178cu };
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) thres_array, "thres_array", 'z', i_vec_thres);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) absmask_array, "absmask_array", 'z', i_vec_absmask);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) scale_array, "scale_array", 'z', i_vec_scale);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) shifter_array, "shifter_array", 'z', i_vec_shifter);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) half_array, "half_array", 'z', i_vec_half);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c0_array, "c0_array", 'z', i_vec_c0);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_array, "c1_array", 'z', i_vec_c1);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_array, "c2_array", 'z', i_vec_c2);
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_ps_minimax3_avx( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   i_gp_reg_tmp,
    const unsigned int                             i_vec_c0_lo,
    const unsigned int                             i_vec_c0_hi,
    const unsigned int                             i_vec_c1_lo,
    const unsigned int                             i_vec_c1_hi,
    const unsigned int                             i_vec_c2_lo,
    const unsigned int                             i_vec_c2_hi,
    int                                            rbp_offs_thres,
    int                                            rbp_offs_signmask,
    int                                            rbp_offs_absmask,
    int                                            rbp_offs_scale,
    int                                            rbp_offs_shifter,
    int                                            rbp_offs_half ) {
  unsigned int thres_array[1] = { 0x40879fff };
  unsigned int signmask_array[1] = { 0x80000000 };
  unsigned int absmask_array[1] = { 0x7fffffff };
  unsigned int scale_array[1] = { 0x406a0ea1 };
  unsigned int shifter_array[1] = { 0x4b400000 };
  unsigned int half_array[1] = { 0x3f000000 };

  unsigned int c0_array[16] = { 0x3ecc4231u, 0x3ecc541cu, 0x3ecd6c48u, 0x3ed174c3u, 0x3ed9bd5du, 0x3ee5acd5u, 0x3ef2aeddu, 0x3efd5384u, 0x3f016724u, 0x3f00f778u, 0x3efb389eu, 0x3ef0464du, 0x3ee3014fu, 0x3ed50a78u, 0x3ec779dbu, 0x3ebae363u };
  unsigned int c1_array[16] = { 0xb7c7fb58u, 0xbacb9740u, 0xbc3e4b3au, 0xbd0d292au, 0xbd8bc5d0u, 0xbdd9978fu, 0xbe0f92d3u, 0xbe27b66du, 0xbe328ce7u, 0xbe3125bfu, 0xbe26dc9du, 0xbe17a056u, 0xbe06bdebu, 0xbdecc593u, 0xbdcf57aau, 0xbdb5ea3au };
  unsigned int c2_array[16] = { 0xbd877b85u, 0xbd7d9780u, 0xbd4cb70eu, 0xbd08a1e9u, 0xbc808857u, 0xb9476fd2u, 0x3c36f765u, 0x3c924160u, 0x3ca7b1fcu, 0x3ca5732cu, 0x3c95af63u, 0x3c8079f7u, 0x3c55fa4fu, 0x3c2fa86bu, 0x3c0fbb00u, 0x3bec178cu };
;

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, thres_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_thres, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, signmask_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_signmask, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, absmask_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_absmask, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, scale_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_scale, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, shifter_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_shifter, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, half_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_half, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c0_array, "c0_array", 'y', i_vec_c0_lo);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) &c0_array[8], "c0_array_", 'y', i_vec_c0_hi);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_array, "c1_array", 'y', i_vec_c1_lo);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) &c1_array[8], "c1_array_", 'y', i_vec_c1_hi);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_array, "c2_array", 'y', i_vec_c2_lo);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) &c2_array[8], "c2_array_", 'y', i_vec_c2_hi);

}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_thres,
    const unsigned int                             i_vec_absmask,
    const unsigned int                             i_vec_scale,
    const unsigned int                             i_vec_shifter,
    const unsigned int                             i_vec_half,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2 ) {
  unsigned int thres_array[16] = { 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff, 0x408f5fff };
  unsigned int absmask_array[16] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
  unsigned int scale_array[16] = { 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9, 0x405d67c9};
  unsigned int shifter_array[16] = { 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000, 0x4b400000 };
  unsigned int half_array[16] = { 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000};
  unsigned int c0_array[16] = { 0x3f4c4245u, 0x3f4c927bu, 0x3f5085f8u, 0x3f5d7bdau, 0x3f73ea12u, 0x3f86142fu, 0x3f8d3df4u, 0x3f8b4b0fu, 0x3f8022c8u, 0x3f5e5423u, 0x3f39ceb5u, 0x3f199bedu, 0x3f00bee0u, 0x3ede1737u, 0x3ec59b86u, 0x3eb4454cu };
  unsigned int c1_array[16] = { 0xb930e738u, 0xbc4b28bau, 0xbda4212fu, 0xbe5feb0eu, 0xbec8b0e5u, 0xbf09e61bu, 0xbf1c403fu, 0xbf185954u, 0xbf03e1eeu, 0xbed08a61u, 0xbe9b4508u, 0xbe61788bu, 0xbe257770u, 0xbdfc542au, 0xbdca014eu, 0xbda8d7e9u };
  unsigned int c2_array[16] = { 0xbe87047bu, 0xbe6eb875u, 0xbe2210c1u, 0xbd81727fu, 0x3cb9625cu, 0x3da2cbe8u, 0x3dd1d4d1u, 0x3dca0bd0u, 0x3da47dd0u, 0x3d6f1bd3u, 0x3d216381u, 0x3cd2618cu, 0x3c89f6e6u, 0x3c3ca672u, 0x3c08ed08u, 0x3bd26a14u };
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) thres_array, "thres_array", 'z', i_vec_thres);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) absmask_array, "absmask_array", 'z', i_vec_absmask);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) scale_array, "scale_array", 'z', i_vec_scale);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) shifter_array, "shifter_array", 'z', i_vec_shifter);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) half_array, "half_array", 'z', i_vec_half);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c0_array, "c0_array", 'z', i_vec_c0);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_array, "c1_array", 'z', i_vec_c1);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_array, "c2_array", 'z', i_vec_c2);
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_gelu_inv_ps_minimax3_avx( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   i_gp_reg_tmp,
    const unsigned int                             i_vec_c0_lo,
    const unsigned int                             i_vec_c0_hi,
    const unsigned int                             i_vec_c1_lo,
    const unsigned int                             i_vec_c1_hi,
    const unsigned int                             i_vec_c2_lo,
    const unsigned int                             i_vec_c2_hi,
    int                                            rbp_offs_thres,
    int                                            rbp_offs_signmask,
    int                                            rbp_offs_absmask,
    int                                            rbp_offs_scale,
    int                                            rbp_offs_shifter,
    int                                            rbp_offs_half ) {

  unsigned int thres_array[1] = { 0x408f5fff };
  unsigned int signmask_array[1] = { 0x80000000 };
  unsigned int absmask_array[1] = { 0x7fffffff };
  unsigned int scale_array[1] = { 0x405d67c9 };
  unsigned int shifter_array[1] = { 0x4b400000 };
  unsigned int half_array[1] = { 0x3f000000 };

  unsigned int c0_array[16] = { 0x3f4c4245u, 0x3f4c927bu, 0x3f5085f8u, 0x3f5d7bdau, 0x3f73ea12u, 0x3f86142fu, 0x3f8d3df4u, 0x3f8b4b0fu, 0x3f8022c8u, 0x3f5e5423u, 0x3f39ceb5u, 0x3f199bedu, 0x3f00bee0u, 0x3ede1737u, 0x3ec59b86u, 0x3eb4454cu };
  unsigned int c1_array[16] = { 0xb930e738u, 0xbc4b28bau, 0xbda4212fu, 0xbe5feb0eu, 0xbec8b0e5u, 0xbf09e61bu, 0xbf1c403fu, 0xbf185954u, 0xbf03e1eeu, 0xbed08a61u, 0xbe9b4508u, 0xbe61788bu, 0xbe257770u, 0xbdfc542au, 0xbdca014eu, 0xbda8d7e9u };
  unsigned int c2_array[16] = { 0xbe87047bu, 0xbe6eb875u, 0xbe2210c1u, 0xbd81727fu, 0x3cb9625cu, 0x3da2cbe8u, 0x3dd1d4d1u, 0x3dca0bd0u, 0x3da47dd0u, 0x3d6f1bd3u, 0x3d216381u, 0x3cd2618cu, 0x3c89f6e6u, 0x3c3ca672u, 0x3c08ed08u, 0x3bd26a14u };

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, thres_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_thres, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, signmask_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_signmask, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, absmask_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_absmask, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, scale_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_scale, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, shifter_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_shifter, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, half_array[0]);
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, rbp_offs_half, i_gp_reg_tmp, 1 );

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c0_array, "c0_array", 'y', i_vec_c0_lo);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) &c0_array[8], "c0_array_", 'y', i_vec_c0_hi);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_array, "c1_array", 'y', i_vec_c1_lo);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) &c1_array[8], "c1_array_", 'y', i_vec_c1_hi);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_array, "c2_array", 'y', i_vec_c2_lo);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) &c2_array[8], "c2_array_", 'y', i_vec_c2_hi);
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vpermd_16way_avx2( libxsmm_generated_code*                        io_generated_code,
                                                const unsigned int                             i_vec_index,
                                                const unsigned int                             i_vec_c_lo,
                                                const unsigned int                             i_vec_c_hi,
                                                const unsigned int                             i_vec_result,
                                                const unsigned int                             i_vec_tmp0,
                                                const unsigned int                             i_vec_tmp1 ) {

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'y',
      i_vec_c_lo,
      i_vec_index,
      i_vec_tmp0,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'y',
      i_vec_c_hi,
      i_vec_index,
      i_vec_tmp1,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'y', i_vec_index, i_vec_result, 28 );

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
            LIBXSMM_X86_INSTR_VBLENDVPS,
            'y',
            i_vec_tmp1,
            i_vec_tmp0,
            i_vec_result,
            0, 0, 0, (unsigned short)((i_vec_result) << 4));
}

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_ps_minimax3_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_c0_lo,
    const unsigned int                             i_vec_c0_hi,
    const unsigned int                             i_vec_c1_lo,
    const unsigned int                             i_vec_c1_hi,
    const unsigned int                             i_vec_c2_lo,
    const unsigned int                             i_vec_c2_hi,
    const unsigned int                             i_vec_tmp0,
    const unsigned int                             i_vec_tmp1,
    const unsigned int                             i_vec_tmp2,
    const unsigned int                             i_vec_tmp3,
    const unsigned int                             i_vec_tmp4,
    const unsigned int                             i_vec_tmp5,
    const unsigned int                             i_vec_tmp6,
    const unsigned int                             i_vec_tmp7,
    int                                            rbp_offs_thres,
    int                                            rbp_offs_signmask,
    int                                            rbp_offs_absmask,
    int                                            rbp_offs_scale,
    int                                            rbp_offs_shifter,
    int                                            rbp_offs_half ) {

  unsigned int i_vec_absmask = i_vec_tmp0, i_vec_xa = i_vec_tmp0;
  unsigned int i_vec_xr = i_vec_tmp1, i_vec_thres = i_vec_tmp1;
  unsigned int i_vec_signmask = i_vec_tmp1;
  unsigned int i_vec_index = i_vec_tmp2;
  unsigned int i_vec_shifter = i_vec_tmp3;
  unsigned int i_vec_scale = i_vec_tmp4;
  unsigned int i_vec_C0 = i_vec_tmp5;
  unsigned int i_vec_C1 = i_vec_tmp6;
  unsigned int i_vec_C2 = i_vec_tmp7;
  unsigned int i_vec_half = i_vec_tmp6;

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_absmask,
                  'y',
                  i_vec_absmask, 0, 0, 0 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VPANDD,
                                       'y',
                                       i_vec_x, i_vec_absmask, i_vec_xa );

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_thres,
                  'y',
                  i_vec_thres, 0, 0, 0 );


  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMINPS, 'y',  i_vec_xa, i_vec_thres, i_vec_xa );
  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', i_vec_xa, i_vec_index );

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_signmask,
                  'y',
                  i_vec_signmask, 0, 0, 0 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VPANDD,
                                       'y',
                                       i_vec_x, i_vec_signmask, i_vec_signmask );


  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VPORD,
                                       'y',
                                       i_vec_signmask, i_vec_xa, i_vec_xr );

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_shifter,
                  'y',
                  i_vec_shifter, 0, 0, 0 );

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_scale,
                  'y',
                  i_vec_scale, 0, 0, 0 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_shifter, i_vec_scale, i_vec_index );


   libxsmm_x86_instruction_vpermd_16way_avx2( io_generated_code,
       i_vec_index,
       i_vec_c0_lo,
       i_vec_c0_hi,
       i_vec_C0,
       i_vec_tmp3,
       i_vec_tmp4);

   libxsmm_x86_instruction_vpermd_16way_avx2( io_generated_code,
       i_vec_index,
       i_vec_c1_lo,
       i_vec_c1_hi,
       i_vec_C1,
       i_vec_tmp3,
       i_vec_tmp4);

   libxsmm_x86_instruction_vpermd_16way_avx2( io_generated_code,
       i_vec_index,
       i_vec_c2_lo,
       i_vec_c2_hi,
       i_vec_C2,
       i_vec_tmp3,
       i_vec_tmp4);

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_C1, i_vec_xa, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_C0, i_vec_xa, i_vec_C2 );

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_half,
                  'y',
                  i_vec_half, 0, 0, 0 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_half, i_vec_xr, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMULPS,
                                       'y',
                                       i_vec_x, i_vec_C2, i_vec_x );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gelu_ps_minimax3_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_xr,
    const unsigned int                             i_vec_xa,
    const unsigned int                             i_vec_index,
    const unsigned int                             i_vec_C0,
    const unsigned int                             i_vec_C1,
    const unsigned int                             i_vec_C2,
    const unsigned int                             i_vec_thres,
    const unsigned int                             i_vec_absmask,
    const unsigned int                             i_vec_scale,
    const unsigned int                             i_vec_shifter,
    const unsigned int                             i_vec_half,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2 ) {

  if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
        LIBXSMM_X86_INSTR_VRANGEPS, 'z',
        i_vec_thres,
        i_vec_x,
        i_vec_xr,
        0, 0, 8, 2);


    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                         LIBXSMM_X86_INSTR_VPANDD,
                                         'z',
                                         i_vec_xr, i_vec_absmask, i_vec_xa );
  } else {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64, 'z', i_vec_x, i_vec_xr );
    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, 'z', i_vec_xr, i_vec_xr, 31 );
    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z', i_vec_xr, i_vec_xr, 31 );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPANDD, 'z', i_vec_x, i_vec_absmask, i_vec_xa );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMINPS, 'z',  i_vec_xa, i_vec_thres, i_vec_xa );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, 'z', i_vec_xr, i_vec_xa, i_vec_xr );
  }

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMOVDQU64,
                                       'z',
                                       i_vec_xa, i_vec_index );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_shifter, i_vec_scale, i_vec_index );

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c2,
      i_vec_index,
      i_vec_C2,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c1,
      i_vec_index,
      i_vec_C1,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c0,
      i_vec_index,
      i_vec_C0,
      0, 0, 0, 0);

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_C1, i_vec_xa, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_C0, i_vec_xa, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_half, i_vec_xr, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMULPS,
                                       'z',
                                       i_vec_x, i_vec_C2, i_vec_x );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_inv_ps_minimax3_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_xr,
    const unsigned int                             i_vec_xa,
    const unsigned int                             i_vec_index,
    const unsigned int                             i_vec_C0,
    const unsigned int                             i_vec_C1,
    const unsigned int                             i_vec_C2,
    const unsigned int                             i_vec_thres,
    const unsigned int                             i_vec_absmask,
    const unsigned int                             i_vec_scale,
    const unsigned int                             i_vec_shifter,
    const unsigned int                             i_vec_half,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2 ) {

  if (io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
        LIBXSMM_X86_INSTR_VRANGEPS, 'z',
        i_vec_thres,
        i_vec_x,
        i_vec_xr,
        0, 0, 8, 2);


    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                         LIBXSMM_X86_INSTR_VPANDD,
                                         'z',
                                         i_vec_xr, i_vec_absmask, i_vec_xa );
  } else {
    libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVDQU64, 'z', i_vec_x, i_vec_xr );
    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, 'z', i_vec_xr, i_vec_xr, 31 );
    libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'z', i_vec_xr, i_vec_xr, 31 );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPANDD, 'z', i_vec_x, i_vec_absmask, i_vec_xa );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMINPS, 'z',  i_vec_xa, i_vec_thres, i_vec_xa );
    libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, 'z', i_vec_xr, i_vec_xa, i_vec_xr );
  }

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMOVDQU64,
                                       'z',
                                       i_vec_xa, i_vec_index );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_shifter, i_vec_scale, i_vec_index );

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c2,
      i_vec_index,
      i_vec_C2,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c1,
      i_vec_index,
      i_vec_C1,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VPERMD, 'z',
      i_vec_c0,
      i_vec_index,
      i_vec_C0,
      0, 0, 0, 0);

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_C1, i_vec_xa, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_C0, i_vec_xa, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_half, i_vec_xr, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMOVDQU64,
                                       'z',
                                       i_vec_C2, i_vec_x );
}

LIBXSMM_API_INTERN
void libxsmm_generator_gelu_inv_ps_minimax3_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_c0_lo,
    const unsigned int                             i_vec_c0_hi,
    const unsigned int                             i_vec_c1_lo,
    const unsigned int                             i_vec_c1_hi,
    const unsigned int                             i_vec_c2_lo,
    const unsigned int                             i_vec_c2_hi,
    const unsigned int                             i_vec_tmp0,
    const unsigned int                             i_vec_tmp1,
    const unsigned int                             i_vec_tmp2,
    const unsigned int                             i_vec_tmp3,
    const unsigned int                             i_vec_tmp4,
    const unsigned int                             i_vec_tmp5,
    const unsigned int                             i_vec_tmp6,
    const unsigned int                             i_vec_tmp7,
    int                                            rbp_offs_thres,
    int                                            rbp_offs_signmask,
    int                                            rbp_offs_absmask,
    int                                            rbp_offs_scale,
    int                                            rbp_offs_shifter,
    int                                            rbp_offs_half ) {

  unsigned int i_vec_absmask = i_vec_tmp0, i_vec_xa = i_vec_tmp0;
  unsigned int i_vec_xr = i_vec_tmp1, i_vec_thres = i_vec_tmp1;
  unsigned int i_vec_signmask = i_vec_tmp1;
  unsigned int i_vec_index = i_vec_tmp2;
  unsigned int i_vec_shifter = i_vec_tmp3;
  unsigned int i_vec_scale = i_vec_tmp4;
  unsigned int i_vec_C0 = i_vec_tmp5;
  unsigned int i_vec_C1 = i_vec_tmp6;
  unsigned int i_vec_C2 = i_vec_tmp7;
  unsigned int i_vec_half = i_vec_tmp6;

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_absmask,
                  'y',
                  i_vec_absmask, 0, 0, 0 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VPANDD,
                                       'y',
                                       i_vec_x, i_vec_absmask, i_vec_xa );

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_thres,
                  'y',
                  i_vec_thres, 0, 0, 0 );


  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMINPS, 'y',  i_vec_xa, i_vec_thres, i_vec_xa );
  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', i_vec_xa, i_vec_index );

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_signmask,
                  'y',
                  i_vec_signmask, 0, 0, 0 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VPANDD,
                                       'y',
                                       i_vec_x, i_vec_signmask, i_vec_signmask );


  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VPORD,
                                       'y',
                                       i_vec_signmask, i_vec_xa, i_vec_xr );

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_shifter,
                  'y',
                  i_vec_shifter, 0, 0, 0 );

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_scale,
                  'y',
                  i_vec_scale, 0, 0, 0 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_shifter, i_vec_scale, i_vec_index );


   libxsmm_x86_instruction_vpermd_16way_avx2( io_generated_code,
       i_vec_index,
       i_vec_c0_lo,
       i_vec_c0_hi,
       i_vec_C0,
       i_vec_tmp3,
       i_vec_tmp4);

   libxsmm_x86_instruction_vpermd_16way_avx2( io_generated_code,
       i_vec_index,
       i_vec_c1_lo,
       i_vec_c1_hi,
       i_vec_C1,
       i_vec_tmp3,
       i_vec_tmp4);

   libxsmm_x86_instruction_vpermd_16way_avx2( io_generated_code,
       i_vec_index,
       i_vec_c2_lo,
       i_vec_c2_hi,
       i_vec_C2,
       i_vec_tmp3,
       i_vec_tmp4);

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_C1, i_vec_xa, i_vec_C2 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_C0, i_vec_xa, i_vec_C2 );

  libxsmm_x86_instruction_unified_vec_move( io_generated_code,
                  LIBXSMM_X86_INSTR_VBROADCASTSS,
                  LIBXSMM_X86_GP_REG_RBP,
                  LIBXSMM_X86_GP_REG_UNDEF, 0,
                  rbp_offs_half,
                  'y',
                  i_vec_half, 0, 0, 0 );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_half, i_vec_xr, i_vec_C2 );

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', i_vec_C2, i_vec_x );
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_exp_ps_3dts_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_halves,
    const unsigned int                             i_vec_log2e ) {
  float c0_array[16] = { 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f };
  float c1_array[16] = { 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f };
  float c2_array[16] = { 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f };
  float c3_array[16] = { 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f };
  float log2e_array[16] = { 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f };
  float halves_array[16] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) halves_array, "halves_array_", 'z', i_vec_halves);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) log2e_array, "log2e_array_", 'z', i_vec_log2e);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c0_array, "c0_array_", 'z', i_vec_c0);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_array, "c1_array_", 'z', i_vec_c1);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_array, "c2_array_", 'z', i_vec_c2);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c3_array, "c3_array_", 'z', i_vec_c3);
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_exp_ps_3dts_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_halves,
    const unsigned int                             i_vec_log2e,
    const unsigned int                             i_vec_expmask,
    const unsigned int                             i_vec_hi_bound,
    const unsigned int                             i_vec_lo_bound ) {
  float c0_array[16] = { 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f, 0.70654502287f };
  float c1_array[16] = { 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f, 0.49454875509f };
  float c2_array[16] = { 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f, 0.15697034396f };
  float c3_array[16] = { 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f, 0.05550410866f };
  float log2e_array[16] = { 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f, 1.442695f };
  float halves_array[16] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
  unsigned int expmask_array[8] = { 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f };
  float hi_b_array[8] = { 88.3762626647949f , 88.3762626647949f, 88.3762626647949f, 88.3762626647949f, 88.3762626647949f, 88.3762626647949f, 88.3762626647949f, 88.3762626647949f };
  float lo_b_array[8] = { -88.3762626647949f, -88.3762626647949f, -88.3762626647949f, -88.3762626647949f, -88.3762626647949f, -88.3762626647949f, -88.3762626647949f, -88.3762626647949f };

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) halves_array, "halves_array_", 'y', i_vec_halves);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) log2e_array, "log2e_array_", 'y', i_vec_log2e);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c0_array, "c0_array_", 'y', i_vec_c0);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_array, "c1_array_", 'y', i_vec_c1);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_array, "c2_array_", 'y', i_vec_c2);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c3_array, "c3_array_", 'y', i_vec_c3);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) expmask_array, "expmask_array_", 'y', i_vec_expmask);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) hi_b_array, "hi_b_array_", 'y', i_vec_hi_bound);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) lo_b_array, "lo_b_array_", 'y', i_vec_lo_bound);
}

LIBXSMM_API_INTERN
void libxsmm_generator_scalefps_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_y,
    const unsigned int                             i_vec_z,
    const unsigned int                             i_vec_expmask ) {
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VROUNDPS, 'y',
      i_vec_y,
      LIBXSMM_X86_VEC_REG_UNDEF,
      i_vec_y,
      0, 0, 0, 1);

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VCVTPS2DQ, 'y',
      i_vec_y,
      LIBXSMM_X86_VEC_REG_UNDEF,
      i_vec_y,
      0, 0, 0, 0);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPADDD, 'y',
                                            i_vec_y, i_vec_expmask, i_vec_y );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, 'y',
                                                 i_vec_y, i_vec_y, 23 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, 'y',
                                            i_vec_x, i_vec_y, i_vec_z );
}

LIBXSMM_API_INTERN
void libxsmm_generator_exp_ps_3dts_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_y,
    const unsigned int                             i_vec_z,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_halves,
    const unsigned int                             i_vec_log2e,
    const unsigned int                             i_vec_expmask,
    const unsigned int                             i_vec_hi_bound,
    const unsigned int                             i_vec_lo_bound ) {

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMAXPS, 'y',  i_vec_x, i_vec_lo_bound, i_vec_x );
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMINPS, 'y',  i_vec_x, i_vec_hi_bound, i_vec_x );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_halves, i_vec_log2e, i_vec_x );

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VROUNDPS, 'y',
      i_vec_x,
      LIBXSMM_X86_VEC_REG_UNDEF,
      i_vec_y,
      0, 0, 0, 1);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, 'y',
                                            i_vec_y, i_vec_x, i_vec_y);

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code, LIBXSMM_X86_INSTR_VMOVUPS, 'y', i_vec_y, i_vec_z );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_c2, i_vec_c3, i_vec_z );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_c1, i_vec_y, i_vec_z );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_c0, i_vec_y, i_vec_z );

  libxsmm_generator_scalefps_avx( io_generated_code, i_vec_z, i_vec_x, i_vec_x, i_vec_expmask);
}


LIBXSMM_API_INTERN
void libxsmm_generator_exp_ps_3dts_avx512( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_y,
    const unsigned int                             i_vec_z,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_halves,
    const unsigned int                             i_vec_log2e ) {

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_halves, i_vec_log2e, i_vec_x );

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VRNDSCALEPS, 'z',
      i_vec_x,
      LIBXSMM_X86_VEC_REG_UNDEF,
      i_vec_y,
      0, 0, 4, 1);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, 'z',
                                            i_vec_y, i_vec_x, i_vec_y);

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMOVDQU64,
                                       'z',
                                       i_vec_y, i_vec_z );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_c2, i_vec_c3, i_vec_z );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_c1, i_vec_y, i_vec_z );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_c0, i_vec_y, i_vec_z );

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code,
      LIBXSMM_X86_INSTR_VSCALEFPS, 'z',
      i_vec_x,
      i_vec_z,
      i_vec_x,
      0, 0, 0, 0);
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_neg_ones ) {
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

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c0_array, "c0_array_", 'z', i_vec_c0);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_array, "c1_array_", 'z', i_vec_c1);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_array, "c2_array_", 'z', i_vec_c2);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c3_array, "c3_array_", 'z', i_vec_c3);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_d_array, "c1_d_array_", 'z', i_vec_c1_d);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_d_array, "c2_d_array_", 'z', i_vec_c2_d);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c3_d_array, "c3_d_array_", 'z', i_vec_c3_d);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) hi_b_array, "hi_b_array_", 'z', i_vec_hi_bound);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) lo_b_array, "lo_b_array_", 'z', i_vec_lo_bound);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) ones_array, "ones_array_", 'z', i_vec_ones);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) neg_ones_array, "neg_ones_array_", 'z', i_vec_neg_ones);
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_avx( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_neg_ones ) {
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

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c0_array, "c0_array_", 'y', i_vec_c0);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_array, "c1_array_", 'y', i_vec_c1);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_array, "c2_array_", 'y', i_vec_c2);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c3_array, "c3_array_", 'y', i_vec_c3);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c1_d_array, "c1_d_array_", 'y', i_vec_c1_d);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c2_d_array, "c2_d_array_", 'y', i_vec_c2_d);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) c3_d_array, "c3_d_array_", 'y', i_vec_c3_d);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) hi_b_array, "hi_b_array_", 'y', i_vec_hi_bound);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) lo_b_array, "lo_b_array_", 'y', i_vec_lo_bound);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) ones_array, "ones_array_", 'y', i_vec_ones);
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) neg_ones_array, "neg_ones_array_", 'y', i_vec_neg_ones);
}

LIBXSMM_API_INTERN
void libxsmm_generator_tanh_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_neg_ones ) {
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        'z',
                                        i_vec_x, i_vec_x, i_vec_x2 );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                LIBXSMM_X86_INSTR_VCMPPS,
                                                'z',
                                                i_vec_hi_bound,
                                                i_vec_x,
                                                i_mask_hi,
                                                17 );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                LIBXSMM_X86_INSTR_VCMPPS,
                                                'z',
                                                i_vec_lo_bound,
                                                i_vec_x,
                                                i_mask_lo,
                                                30 );

   libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMOVDQU64,
                                       'z',
                                       i_vec_x2, i_vec_nom );

   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_c2, i_vec_c3, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_c1, i_vec_x2, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_c0, i_vec_x2, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        'z',
                                        i_vec_nom, i_vec_x, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VADDPS,
                                       'z',
                                       i_vec_x2, i_vec_c3_d, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_c2_d, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_c1_d, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'z',
                                       i_vec_c0, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VRCP14PS,
                                       'z',
                                       i_vec_denom, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        'z',
                                        i_vec_denom, i_vec_nom, i_vec_x );

  libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                LIBXSMM_X86_INSTR_VBLENDMPS,
                                                'z',
                                                i_vec_x,
                                                i_vec_ones,
                                                i_vec_x,
                                                i_mask_hi, 0 );

  libxsmm_x86_instruction_vec_compute_3reg_mask( io_generated_code,
                                                LIBXSMM_X86_INSTR_VBLENDMPS,
                                                'z',
                                                i_vec_x,
                                                i_vec_neg_ones,
                                                i_vec_x,
                                                i_mask_lo, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_tanh_ps_rational_78_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_x2,
    const unsigned int                             i_vec_nom,
    const unsigned int                             i_vec_denom,
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
    const unsigned int                             i_vec_neg_ones ) {

  unsigned int i_mask_hi = i_vec_nom;
  unsigned int i_mask_lo = i_vec_denom;

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        'y',
                                        i_vec_x, i_vec_x, i_vec_x2 );


   libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMOVUPS, 'y',i_vec_x2, i_vec_nom );


   libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_c2, i_vec_c3, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_c1, i_vec_x2, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_c0, i_vec_x2, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        'y',
                                        i_vec_nom, i_vec_x, i_vec_nom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VADDPS,
                                       'y',
                                       i_vec_x2, i_vec_c3_d, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_c2_d, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_c1_d, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VFMADD213PS,
                                       'y',
                                       i_vec_c0, i_vec_x2, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VRCPPS,
                                       'y',
                                       i_vec_denom, i_vec_denom );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                        LIBXSMM_X86_INSTR_VMULPS,
                                        'y',
                                        i_vec_denom, i_vec_nom, i_vec_x2 );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                LIBXSMM_X86_INSTR_VCMPPS,
                                                'y',
                                                i_vec_hi_bound,
                                                i_vec_x,
                                                i_mask_hi,
                                                17 );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                                LIBXSMM_X86_INSTR_VCMPPS,
                                                'y',
                                                i_vec_lo_bound,
                                                i_vec_x,
                                                i_mask_lo,
                                                30 );

  libxsmm_x86_instruction_vec_compute_2reg( io_generated_code,
                                       LIBXSMM_X86_INSTR_VMOVUPS, 'y',i_vec_x2, i_vec_x );

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
            LIBXSMM_X86_INSTR_VBLENDVPS,
            'y',
            i_vec_x,
            i_vec_ones,
            i_vec_x,
            0, 0, 0, (unsigned short)((i_mask_hi) << 4));

  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8(io_generated_code,
            LIBXSMM_X86_INSTR_VBLENDVPS,
            'y',
            i_vec_x,
            i_vec_neg_ones,
            i_vec_x,
            0, 0, 0, (unsigned short)((i_mask_lo) << 4));
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_neg_ones,
    const unsigned int                             i_vec_halves ) {
  float halves_array[16] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) halves_array, "halves_array_", 'z', i_vec_halves);
  libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_avx512( io_generated_code, i_vec_c0, i_vec_c1, i_vec_c2, i_vec_c3, i_vec_c1_d, i_vec_c2_d, i_vec_c3_d, i_vec_hi_bound, i_vec_lo_bound, i_vec_ones, i_vec_neg_ones );
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_coeffs_sigmoid_ps_rational_78_avx( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_neg_ones) {
  libxsmm_generator_prepare_coeffs_tanh_ps_rational_78_avx( io_generated_code, i_vec_c0, i_vec_c1, i_vec_c2, i_vec_c3, i_vec_c1_d, i_vec_c2_d, i_vec_c3_d, i_vec_hi_bound, i_vec_lo_bound, i_vec_ones, i_vec_neg_ones );
}


LIBXSMM_API_INTERN
void libxsmm_generator_load_prng_state_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                   const unsigned char     i_vname,
                                                   const unsigned int      i_gp_reg_prng_state_ptr,
                                                   const unsigned int      prng_state0_vreg,
                                                   const unsigned int      prng_state1_vreg,
                                                   const unsigned int      prng_state2_vreg,
                                                   const unsigned int      prng_state3_vreg ) {
  /* load RNG state */
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VMOVUPS_LD, i_gp_reg_prng_state_ptr, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_vname, prng_state0_vreg, 0, 1, 0 );
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VMOVUPS_LD, i_gp_reg_prng_state_ptr, LIBXSMM_X86_GP_REG_UNDEF, 0, 64,
                                    i_vname, prng_state1_vreg, 0, 1, 0 );
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VMOVUPS_LD, i_gp_reg_prng_state_ptr, LIBXSMM_X86_GP_REG_UNDEF, 0, 128,
                                    i_vname, prng_state2_vreg, 0, 1, 0 );
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VMOVUPS_LD, i_gp_reg_prng_state_ptr, LIBXSMM_X86_GP_REG_UNDEF, 0, 192,
                                    i_vname, prng_state3_vreg, 0, 1, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_store_prng_state_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                    const unsigned char     i_vname,
                                                    const unsigned int      i_gp_reg_prng_state_ptr,
                                                    const unsigned int      prng_state0_vreg,
                                                    const unsigned int      prng_state1_vreg,
                                                    const unsigned int      prng_state2_vreg,
                                                    const unsigned int      prng_state3_vreg ) {
  /* load RNG state */
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VMOVUPS_ST, i_gp_reg_prng_state_ptr, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_vname, prng_state0_vreg, 0, 0, 1 );
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VMOVUPS_ST, i_gp_reg_prng_state_ptr, LIBXSMM_X86_GP_REG_UNDEF, 0, 64,
                                    i_vname, prng_state1_vreg, 0, 0, 1 );
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VMOVUPS_ST, i_gp_reg_prng_state_ptr, LIBXSMM_X86_GP_REG_UNDEF, 0, 128,
                                    i_vname, prng_state2_vreg, 0, 0, 1 );
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VMOVUPS_ST, i_gp_reg_prng_state_ptr, LIBXSMM_X86_GP_REG_UNDEF, 0, 192,
                                    i_vname, prng_state3_vreg, 0, 0, 1 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_load_vreg_minus_infinity(libxsmm_generated_code* io_generated_code,
                                                   const unsigned char     i_vname,
                                                   const unsigned int      i_gp_reg_tmp,
                                                   const unsigned int      i_vreg_minus_infinity) {
  /* load constant register with minus infinity */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, 0xff800000);
  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_tmp );
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_vname, i_vreg_minus_infinity, 0, 1, 0 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_tmp );
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_dropout_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                   const unsigned char     i_vname,
                                                   const unsigned int      i_gp_reg_tmp,
                                                   const unsigned int      i_gp_reg_prob_ptr,
                                                   const unsigned int      dropout_vreg_one,
                                                   const unsigned int      dropout_prob_vreg,
                                                   const unsigned int      dropout_invprob_vreg ) {
  /* load constant register */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, 0x3f800000);
  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_tmp );
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_vname, dropout_vreg_one, 0, 1, 0 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_tmp );

  /* load probability */
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    i_gp_reg_prob_ptr, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_vname, dropout_prob_vreg, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, i_vname,
                                            dropout_prob_vreg, dropout_vreg_one, dropout_prob_vreg );

  /* load 1/prob */
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VDIVPS, i_vname,
                                            dropout_prob_vreg, dropout_vreg_one, dropout_invprob_vreg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_prepare_dropout_inv_avx_avx512( libxsmm_generated_code* io_generated_code,
                                                       const unsigned char     i_vname,
                                                       const unsigned int      i_gp_reg_tmp,
                                                       const unsigned int      i_gp_reg_prob_ptr,
                                                       const unsigned int      dropout_vreg_one,
                                                       const unsigned int      dropout_vreg_zero,
                                                       const unsigned int      dropout_prob_vreg ) {

  /* load constant register */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_tmp, 0x3f800000);
  libxsmm_x86_instruction_push_reg( io_generated_code, i_gp_reg_tmp );
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_vname, dropout_vreg_one, 0, 1, 0 );
  libxsmm_x86_instruction_pop_reg( io_generated_code, i_gp_reg_tmp );

  /* load probability */
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch,
                                    LIBXSMM_X86_INSTR_VBROADCASTSS,
                                    i_gp_reg_prob_ptr, LIBXSMM_X86_GP_REG_UNDEF, 0, 0,
                                    i_vname, dropout_prob_vreg, 0, 1, 0 );
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, i_vname,
                                            dropout_prob_vreg, dropout_vreg_one, dropout_prob_vreg );

  /* load 1/prob */
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VDIVPS, i_vname,
                                            dropout_prob_vreg, dropout_vreg_one, dropout_prob_vreg );

  /* load zero, for masking */
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            dropout_vreg_zero, dropout_vreg_zero, dropout_vreg_zero );
}

LIBXSMM_API_INTERN
void libxsmm_generator_sigmoid_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
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
    const unsigned int                             i_vec_neg_ones,
    const unsigned int                             i_vec_halves ) {
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, 'z', i_vec_x, i_vec_halves, i_vec_x );
  libxsmm_generator_tanh_ps_rational_78_avx512( io_generated_code, i_vec_x, i_vec_x2, i_vec_nom, i_vec_denom, i_mask_hi, i_mask_lo, i_vec_c0, i_vec_c1, i_vec_c2, i_vec_c3, i_vec_c1_d, i_vec_c2_d, i_vec_c3_d, i_vec_hi_bound, i_vec_lo_bound, i_vec_ones, i_vec_neg_ones );
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, 'z', i_vec_x, i_vec_ones, i_vec_x );
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, 'z', i_vec_x, i_vec_halves, i_vec_x );
}

LIBXSMM_API_INTERN
void libxsmm_generator_sigmoid_ps_rational_78_avx( libxsmm_generated_code*                        io_generated_code,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_x2,
    const unsigned int                             i_vec_nom,
    const unsigned int                             i_vec_denom,
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
    const unsigned int                             i_vec_neg_ones) {

  unsigned int i_vec_halves = i_vec_x2;
  float halves_array[16] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) halves_array, "halves_array_", 'y', i_vec_halves);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, 'y', i_vec_x, i_vec_halves, i_vec_x );
  libxsmm_generator_tanh_ps_rational_78_avx( io_generated_code, i_vec_x, i_vec_x2, i_vec_nom, i_vec_denom, i_vec_c0, i_vec_c1, i_vec_c2, i_vec_c3, i_vec_c1_d, i_vec_c2_d, i_vec_c3_d, i_vec_hi_bound, i_vec_lo_bound, i_vec_ones, i_vec_neg_ones );

  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) halves_array, "halves_array_", 'y', i_vec_halves);
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VADDPS, 'y', i_vec_x, i_vec_ones, i_vec_x );
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, 'y', i_vec_x, i_vec_halves, i_vec_x );
}

LIBXSMM_API_INTERN
void libxsmm_generator_hinstrps_avx( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   instr,
    const unsigned int                             i_vec_inout,
    const unsigned int                             i_vec_tmp1,
    const unsigned int                             i_vec_tmp2) {

  if (i_vec_tmp1 > 15 || i_vec_tmp2 > 15 ) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                           LIBXSMM_X86_INSTR_VPERM2F128,
                                           'y',
                                           i_vec_inout, i_vec_inout, i_vec_tmp1, 0x1 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           instr,
                                           'y',
                                           i_vec_inout, i_vec_tmp1, i_vec_tmp2 );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp2, i_vec_tmp1, 0x4e );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           instr,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp1, i_vec_tmp2 );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp2, i_vec_tmp1, 0x1 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           instr,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp1, i_vec_inout );
}

LIBXSMM_API_INTERN
void libxsmm_generator_hinstrps_avx512( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   instr,
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
                                           instr,
                                           'z',
                                           i_vec_inout, i_vec_tmp1, i_vec_inout );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                           LIBXSMM_X86_INSTR_VSHUFF64X2,
                                           'z',
                                           i_vec_inout, i_vec_inout, i_vec_tmp1, 0xb1 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           instr,
                                           'z',
                                           i_vec_inout, i_vec_tmp1, i_vec_tmp2 );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp2, i_vec_tmp1, 0x4e );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           instr,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp1, i_vec_tmp2 );

  libxsmm_x86_instruction_vec_compute_3reg_imm8( io_generated_code,
                                           LIBXSMM_X86_INSTR_VSHUFPS,
                                           'y',
                                           i_vec_tmp2, i_vec_tmp2, i_vec_tmp1, 0x1 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code,
                                           instr,
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
void libxsmm_generator_generic_loop_header_no_idx_inc( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const unsigned int                 i_loop_reg,
    const unsigned int                 i_loop_init_val) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_loop_reg, i_loop_init_val );
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_generic_loop_footer_with_idx_inc( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const unsigned int                 i_loop_reg,
    const unsigned int                 i_loop_step,
    const unsigned int                 i_loop_bound) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_loop_reg, i_loop_step);
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_CMPQ, i_loop_reg, i_loop_bound );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, LIBXSMM_X86_INSTR_JL, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128pp_axv2_avx512( libxsmm_generated_code* io_generated_code,
                                                 const unsigned char     i_vname,
                                                 const unsigned int      i_vec_reg_rng_state_0,
                                                 const unsigned int      i_vec_reg_rng_state_1,
                                                 const unsigned int      i_vec_reg_rng_state_2,
                                                 const unsigned int      i_vec_reg_rng_state_3,
                                                 const unsigned int      i_vec_reg_rng_tmp_0,
                                                 const unsigned int      i_vec_reg_rng_tmp_1,
                                                 const unsigned int      o_vec_reg_rng ) {
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPADDD, i_vname,
                                            i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, i_vec_reg_rng_tmp_0 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_vname,
                                                 i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, 7 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, i_vname,
                                                 i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_0, 25 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, i_vname,
                                            i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, i_vec_reg_rng_tmp_0 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPADDD, i_vname,
                                            i_vec_reg_rng_tmp_0, i_vec_reg_rng_state_0, o_vec_reg_rng);

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_vname,
                                                 i_vec_reg_rng_state_1, i_vec_reg_rng_tmp_0, 9);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            i_vec_reg_rng_state_2, i_vec_reg_rng_state_0, i_vec_reg_rng_state_2 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            i_vec_reg_rng_state_3, i_vec_reg_rng_state_1, i_vec_reg_rng_state_3 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            i_vec_reg_rng_state_1, i_vec_reg_rng_state_2, i_vec_reg_rng_state_1 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, i_vec_reg_rng_state_0 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            i_vec_reg_rng_state_2, i_vec_reg_rng_tmp_0, i_vec_reg_rng_state_2 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_vname,
                                                 i_vec_reg_rng_state_3, i_vec_reg_rng_tmp_0, 11 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, i_vname,
                                                 i_vec_reg_rng_state_3, i_vec_reg_rng_tmp_1, 21 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, i_vname,
                                            i_vec_reg_rng_tmp_0, i_vec_reg_rng_tmp_1, i_vec_reg_rng_state_3);
}

LIBXSMM_API_INTERN
void libxsmm_generator_xoshiro128p_f32_avx2_avx512( libxsmm_generated_code* io_generated_code,
                                                    const unsigned char     i_vname,
                                                    const unsigned int      i_vec_reg_rng_state_0,
                                                    const unsigned int      i_vec_reg_rng_state_1,
                                                    const unsigned int      i_vec_reg_rng_state_2,
                                                    const unsigned int      i_vec_reg_rng_state_3,
                                                    const unsigned int      i_vec_reg_rng_tmp_0,
                                                    const unsigned int      i_vec_reg_rng_tmp_1,
                                                    const unsigned int      i_vec_reg_rng_one,
                                                    const unsigned int      o_vec_reg_rng ) {
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPADDD, i_vname,
                                            i_vec_reg_rng_state_3, i_vec_reg_rng_state_0, o_vec_reg_rng);

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, i_vname,
                                                 o_vec_reg_rng, o_vec_reg_rng, 9);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, i_vname,
                                            o_vec_reg_rng, i_vec_reg_rng_one, o_vec_reg_rng);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VSUBPS, i_vname,
                                            i_vec_reg_rng_one, o_vec_reg_rng, o_vec_reg_rng);

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_vname,
                                                 i_vec_reg_rng_state_1, i_vec_reg_rng_tmp_0, 9);

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            i_vec_reg_rng_state_2, i_vec_reg_rng_state_0, i_vec_reg_rng_state_2 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            i_vec_reg_rng_state_3, i_vec_reg_rng_state_1, i_vec_reg_rng_state_3 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            i_vec_reg_rng_state_1, i_vec_reg_rng_state_2, i_vec_reg_rng_state_1 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            i_vec_reg_rng_state_0, i_vec_reg_rng_state_3, i_vec_reg_rng_state_0 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPXORD, i_vname,
                                            i_vec_reg_rng_state_2, i_vec_reg_rng_tmp_0, i_vec_reg_rng_state_2 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSLLD_I, i_vname,
                                                 i_vec_reg_rng_state_3, i_vec_reg_rng_tmp_0, 11 );

  libxsmm_x86_instruction_vec_compute_2reg_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPSRLD_I, i_vname,
                                                 i_vec_reg_rng_state_3, i_vec_reg_rng_tmp_1, 21 );

  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VPORD, i_vname,
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
void libxsmm_generator_vcvtneps2bf16_avx512_preppedstack_nocompact( libxsmm_generated_code* io_generated_code,
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
void libxsmm_generator_vcvtneps2int_avx512( libxsmm_generated_code* io_generated_code,
                                            const libxsmm_datatype  i_datatype,
                                            const unsigned int      io_vec_reg,
                                            const unsigned int      i_scf_vec_reg ) {
  /* scale value */
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, 'z', io_vec_reg, i_scf_vec_reg, io_vec_reg );

  /* convert to int */
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTPS2DQ, 'z', io_vec_reg, LIBXSMM_X86_VEC_REG_UNDEF, io_vec_reg, 0, 0, 0, 0);

  if ( i_datatype == LIBXSMM_DATATYPE_I16 ) {
    /* @TODO: add rouding */
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPMOVDW, 'z', io_vec_reg, LIBXSMM_X86_VEC_REG_UNDEF, io_vec_reg, 0, 0, 0, 0);
  } else if ( i_datatype == LIBXSMM_DATATYPE_I8 ) {
    /* @TODO: add rouding */
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPMOVDB, 'z', io_vec_reg, LIBXSMM_X86_VEC_REG_UNDEF, io_vec_reg, 0, 0, 0, 0);
  } else if ( i_datatype == LIBXSMM_DATATYPE_I32 ) {
    /* nothing to do */
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_vcvtint2ps_avx512( libxsmm_generated_code* io_generated_code,
                                          const libxsmm_datatype  i_datatype,
                                          const unsigned int      io_vec_reg,
                                          const unsigned int      i_scf_vec_reg ) {
  if ( i_datatype == LIBXSMM_DATATYPE_I16 ) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXWD, 'z', io_vec_reg, LIBXSMM_X86_VEC_REG_UNDEF, io_vec_reg, 0, 0, 0, 0);
  } else if ( i_datatype == LIBXSMM_DATATYPE_I8 ) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VPMOVSXBD, 'z', io_vec_reg, LIBXSMM_X86_VEC_REG_UNDEF, io_vec_reg, 0, 0, 0, 0);
  } else if ( i_datatype == LIBXSMM_DATATYPE_I32 ) {
    /* nothing to do */
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  /* convert to int */
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( io_generated_code, LIBXSMM_X86_INSTR_VCVTDQ2PS, 'z', io_vec_reg, LIBXSMM_X86_VEC_REG_UNDEF, io_vec_reg, 0, 0, 0, 0);

  /* scale value */
  libxsmm_x86_instruction_vec_compute_3reg( io_generated_code, LIBXSMM_X86_INSTR_VMULPS, 'z', io_vec_reg, i_scf_vec_reg, io_vec_reg );
}

