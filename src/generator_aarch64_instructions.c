/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_aarch64_instructions.h"

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_open_stream( libxsmm_generated_code* io_generated_code,
                                              const unsigned short    i_callee_save_bitmask ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_close_stream: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  /* allocate callee save space on the stack */
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP,
                                                 192, 0 );

  /* save lower 64bit of v8-v15 to stack */
  if ( ( i_callee_save_bitmask & 0x1 ) == 0x1 ) {
    libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 176,
                                                 LIBXSMM_AARCH64_ASIMD_REG_V8, LIBXSMM_AARCH64_ASIMD_REG_V9,
                                                 LIBXSMM_AARCH64_ASIMD_WIDTH_D );
  }
  if ( ( i_callee_save_bitmask & 0x2 ) == 0x2 ) {
    libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 160,
                                                 LIBXSMM_AARCH64_ASIMD_REG_V10, LIBXSMM_AARCH64_ASIMD_REG_V11,
                                                 LIBXSMM_AARCH64_ASIMD_WIDTH_D );
  }
  if ( ( i_callee_save_bitmask & 0x4 ) == 0x4 ) {
    libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 144,
                                                 LIBXSMM_AARCH64_ASIMD_REG_V12, LIBXSMM_AARCH64_ASIMD_REG_V13,
                                                 LIBXSMM_AARCH64_ASIMD_WIDTH_D );
  }
  if ( ( i_callee_save_bitmask & 0x8 ) == 0x8 ) {
    libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 128,
                                                 LIBXSMM_AARCH64_ASIMD_REG_V14, LIBXSMM_AARCH64_ASIMD_REG_V15,
                                                 LIBXSMM_AARCH64_ASIMD_WIDTH_D );
  }

  /* save x16-x30 to stack */
  if ( ( i_callee_save_bitmask & 0x10 ) == 0x10 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 112,
                                               LIBXSMM_AARCH64_GP_REG_X16, LIBXSMM_AARCH64_GP_REG_X17 );
  }
  if ( ( i_callee_save_bitmask & 0x20 ) == 0x20 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  96,
                                               LIBXSMM_AARCH64_GP_REG_X18, LIBXSMM_AARCH64_GP_REG_X19 );
  }
  if ( ( i_callee_save_bitmask & 0x40 ) == 0x40 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  80,
                                               LIBXSMM_AARCH64_GP_REG_X20, LIBXSMM_AARCH64_GP_REG_X21 );
  }
  if ( ( i_callee_save_bitmask & 0x80 ) == 0x80 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  64,
                                               LIBXSMM_AARCH64_GP_REG_X22, LIBXSMM_AARCH64_GP_REG_X23 );
  }
  if ( ( i_callee_save_bitmask & 0x100 ) == 0x100 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  48,
                                               LIBXSMM_AARCH64_GP_REG_X24, LIBXSMM_AARCH64_GP_REG_X25 );
  }
  if ( ( i_callee_save_bitmask & 0x200 ) == 0x200 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  32,
                                               LIBXSMM_AARCH64_GP_REG_X26, LIBXSMM_AARCH64_GP_REG_X27 );
  }
  if ( ( i_callee_save_bitmask & 0x400 ) == 0x400 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  16,
                                               LIBXSMM_AARCH64_GP_REG_X28, LIBXSMM_AARCH64_GP_REG_X29 );
  }
  if ( ( i_callee_save_bitmask & 0x800 ) == 0x800 ) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR, 0,
                                          LIBXSMM_AARCH64_GP_REG_X30 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_close_stream( libxsmm_generated_code* io_generated_code,
                                               const unsigned short    i_callee_save_bitmask ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_close_stream: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  /* restor lower 64bit of v8-v15 from stack */
  if ( ( i_callee_save_bitmask & 0x1 ) == 0x1 ) {
    libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 176,
                                                 LIBXSMM_AARCH64_ASIMD_REG_V8, LIBXSMM_AARCH64_ASIMD_REG_V9,
                                                 LIBXSMM_AARCH64_ASIMD_WIDTH_D );
  }
  if ( ( i_callee_save_bitmask & 0x2 ) == 0x2 ) {
    libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 160,
                                                 LIBXSMM_AARCH64_ASIMD_REG_V10, LIBXSMM_AARCH64_ASIMD_REG_V11,
                                                 LIBXSMM_AARCH64_ASIMD_WIDTH_D );
  }
  if ( ( i_callee_save_bitmask & 0x4 ) == 0x4 ) {
    libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 144,
                                                 LIBXSMM_AARCH64_ASIMD_REG_V12, LIBXSMM_AARCH64_ASIMD_REG_V13,
                                                 LIBXSMM_AARCH64_ASIMD_WIDTH_D );
  }
  if ( ( i_callee_save_bitmask & 0x8 ) == 0x8 ) {
    libxsmm_aarch64_instruction_asimd_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 128,
                                                 LIBXSMM_AARCH64_ASIMD_REG_V14, LIBXSMM_AARCH64_ASIMD_REG_V15,
                                                 LIBXSMM_AARCH64_ASIMD_WIDTH_D );
  }

  /* restor x16-x30 from stack */
  if ( ( i_callee_save_bitmask & 0x10 ) == 0x10 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 112,
                                               LIBXSMM_AARCH64_GP_REG_X16, LIBXSMM_AARCH64_GP_REG_X17 );
  }
  if ( ( i_callee_save_bitmask & 0x20 ) == 0x20 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  96,
                                               LIBXSMM_AARCH64_GP_REG_X18, LIBXSMM_AARCH64_GP_REG_X19 );
  }
  if ( ( i_callee_save_bitmask & 0x40 ) == 0x40 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  80,
                                               LIBXSMM_AARCH64_GP_REG_X20, LIBXSMM_AARCH64_GP_REG_X21 );
  }
  if ( ( i_callee_save_bitmask & 0x80 ) == 0x80 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  64,
                                               LIBXSMM_AARCH64_GP_REG_X22, LIBXSMM_AARCH64_GP_REG_X23 );
  }
  if ( ( i_callee_save_bitmask & 0x100 ) == 0x100 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  48,
                                               LIBXSMM_AARCH64_GP_REG_X24, LIBXSMM_AARCH64_GP_REG_X25 );
  }
  if ( ( i_callee_save_bitmask & 0x200 ) == 0x200 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  32,
                                               LIBXSMM_AARCH64_GP_REG_X26, LIBXSMM_AARCH64_GP_REG_X27 );
  }
  if ( ( i_callee_save_bitmask & 0x400 ) == 0x400 ) {
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP,  16,
                                               LIBXSMM_AARCH64_GP_REG_X28, LIBXSMM_AARCH64_GP_REG_X29 );
  }
  if ( ( i_callee_save_bitmask & 0x800 ) == 0x800 ) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XZR, 0,
                                          LIBXSMM_AARCH64_GP_REG_X30 );
  }

  /* deallocate calle save space on stack */
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I,
                                                 LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP,
                                                 192, 0 );

  /* generate return instruction */
  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* insert ret instruction */
    code[code_head] = 0xd65f03c0;

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_close_stream: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_move( libxsmm_generated_code*           io_generated_code,
                                             const unsigned int                i_vmove_instr,
                                             const unsigned char               i_gp_reg_addr,
                                             const unsigned char               i_gp_reg_offset,
                                             const short                       i_offset,
                                             const unsigned char               i_vec_reg,
                                             const libxsmm_aarch64_asimd_width i_asimdwidth ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_move: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_vmove_instr ) {
    case LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_PRE:
    case LIBXSMM_AARCH64_INSTR_ASIMD_STR_R:
    case LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF:
    case LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST:
    case LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_PRE:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_asimd_move: unexpected instruction number: %u\n", i_vmove_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int*)io_generated_code->generated_code;

    /* fix bits */
    code[code_head] = (unsigned int)(0xffffff00 & i_vmove_instr);
    /* setting Rt */
    code[code_head] |= (unsigned int)(0x1f & i_vec_reg);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_addr) << 5);
   /* setting size */
    code[code_head] |= (unsigned int)((0x6 & (unsigned int)i_asimdwidth) << 29);
    /* setting opc */
    code[code_head] |= (unsigned int)((0x1 & (unsigned int)i_asimdwidth) << 23);

    /* load/store with offset register */
    if ( ((i_vmove_instr & 0x3) == 0x3) && ((i_vmove_instr & 0x4) == 0x0) ) {
      /* setting Rm */
      code[code_head] |= (unsigned int)((0x1f & i_gp_reg_offset) << 16);
      /* setting lowest option bit based on the register name */
      code[code_head] |= (unsigned int)((0x20 & i_gp_reg_offset) << 8);
    }

    /* load/store with imm offset */
    if ( (i_vmove_instr & 0x6) == 0x6 ) {
      /* set imm */
      if ( i_vmove_instr == LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF ||
           i_vmove_instr == LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF     ) {
        short l_imm;
        /* adjust offset based on vector width */
        switch ( i_asimdwidth ) {
          case LIBXSMM_AARCH64_ASIMD_WIDTH_H:
            l_imm = i_offset/2;
            break;
          case LIBXSMM_AARCH64_ASIMD_WIDTH_S:
            l_imm = i_offset/4;
            break;
          case LIBXSMM_AARCH64_ASIMD_WIDTH_D:
            l_imm = i_offset/8;
            break;
          case LIBXSMM_AARCH64_ASIMD_WIDTH_Q:
            l_imm = i_offset/16;
            break;
          default:
            l_imm = i_offset;
            break;
        }
        if ( (l_imm > 0x0fff) || (i_offset < 0) ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_asimd_move: offset for unsigned offnset addressing mode out of range: %i, %i!\n", l_imm, i_offset);
          exit(-1);
        }
        code[code_head] |= (unsigned int)((0x00000fff & l_imm) << 10);
      } else {
        if ( (i_offset < -256) || (i_offset > 255) ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_asimd_move: offset for per-index/post-index addressing mode out of range: %i!\n", i_offset);
          exit(-1);
        }
        code[code_head] |= (unsigned int)((0x000001ff & i_offset) << 12);
      }
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_move: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_struct_move( libxsmm_generated_code*                io_generated_code,
                                                    const unsigned int                     i_vmove_instr,
                                                    const unsigned char                    i_gp_reg_addr,
                                                    const unsigned char                    i_gp_reg_offset,
                                                    const unsigned char                    i_vec_reg,
                                                    const libxsmm_aarch64_asimd_structtype i_structtype ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_move: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_vmove_instr ) {
    case LIBXSMM_AARCH64_INSTR_ASIMD_LD1R:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_move: unexpected instruction number: %u\n", i_vmove_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int*)io_generated_code->generated_code;

    /* fix bits */
    code[code_head] = (unsigned int)(0xffffff00 & i_vmove_instr);
    /* setting Rt */
    code[code_head] |= (unsigned int)(0x1f & i_vec_reg);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_addr) << 5);
   /* setting size */
    code[code_head] |= (unsigned int)((0x6 & (unsigned int)i_structtype) << 9);
    /* setting opc */
    code[code_head] |= (unsigned int)((0x1 & (unsigned int)i_structtype) << 30);

    /* load/store with offset register */
    if ( (i_vmove_instr & 0x3) == 0x3 ) {
      /* setting Rm */
      code[code_head] |= (unsigned int)((0x1f & i_gp_reg_offset) << 16);
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_move: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_pair_move( libxsmm_generated_code*           io_generated_code,
                                                  const unsigned int                i_vmove_instr,
                                                  const unsigned char               i_gp_reg_addr,
                                                  const short                       i_offset,
                                                  const unsigned char               i_vec_reg_0,
                                                  const unsigned char               i_vec_reg_1,
                                                  const libxsmm_aarch64_asimd_width i_asimdwidth ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_pair_move: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_vmove_instr ) {
    case LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_POST:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_PRE:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LDNP_I_OFF:
    case LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF:
    case LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_POST:
    case LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_PRE:
    case LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_asimd_pair_move: unexpected instruction number: %u\n", i_vmove_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;
    unsigned char l_opc = 0x0;
    signed char l_imm = 0x0;

    switch ( i_asimdwidth ) {
      case LIBXSMM_AARCH64_ASIMD_WIDTH_S:
        l_opc = 0x0;
        l_imm = (signed char)(i_offset/4);
        break;
      case LIBXSMM_AARCH64_ASIMD_WIDTH_D:
        l_opc = 0x1;
        l_imm = (signed char)(i_offset/8);
        break;
      case LIBXSMM_AARCH64_ASIMD_WIDTH_Q:
        l_opc = 0x2;
        l_imm = (signed char)(i_offset/16);
        break;
      default:
        fprintf(stderr, "libxsmm_aarch64_instruction_asimd_pair_move: unexpected asimdwidth number: %u\n", i_asimdwidth);
        exit(-1);
    }

    if ( (l_imm < -64) || (l_imm > 63) ) {
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: offset out of range: %i!\n", i_offset);
      exit(-1);
    }

    /* fix bits */
    code[code_head] = (unsigned int)(0xffffff00 & i_vmove_instr);
    /* setting Rt */
    code[code_head] |= (unsigned int)(0x1f & i_vec_reg_0);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_addr) << 5);
    /* setting Rt2 */
    code[code_head] |= (unsigned int)((0x1f & i_vec_reg_1) << 10);
    /* setting imm7 */
    code[code_head] |= (unsigned int)((0x7f & l_imm) << 15);
    /* set opc */
    code[code_head] |= (unsigned int)(l_opc << 30);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_pair_move: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_compute( libxsmm_generated_code*               io_generated_code,
                                                const unsigned int                    i_vec_instr,
                                                const unsigned char                   i_vec_reg_src_0,
                                                const unsigned char                   i_vec_reg_src_1,
                                                const unsigned char                   i_index,
                                                const unsigned char                   i_vec_reg_dst,
                                                const libxsmm_aarch64_asimd_tupletype i_tupletype ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_compute: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_vec_instr ) {
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_S:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_asimd_compute: unexpected instruction number: %u\n", i_vec_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* fix bits */
    code[code_head]  = (unsigned int)(0xffffff00 & i_vec_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)(0x1f & i_vec_reg_dst);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_vec_reg_src_0) << 5);
    /* setting Rm */
    code[code_head] |= (unsigned int)((0x1f & i_vec_reg_src_1) << 16);
    /* setting Q */
    code[code_head] |= (unsigned int)((0x2 & i_tupletype) << 29);
    /* setting sz */
    code[code_head] |= (unsigned int)((0x1 & i_tupletype) << 22);
    if ( (i_vec_instr == LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_S) ||
         (i_vec_instr == LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V)    ) {
      unsigned char l_idx = ( i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D ) ? i_index << 1 : i_index;
      if ( (i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D && i_index > 2) || (i_index > 4) ) {
        fprintf(stderr, "libxsmm_aarch64_instruction_asimd_compute: inoompatible tuple and index type for instruction: %u\n", i_vec_instr);
        exit(-1);
      }

      /* setting L */
      code[code_head] |= (unsigned int)((0x1 & l_idx) << 21);
      /* setting H */
      code[code_head] |= (unsigned int)((0x2 & l_idx) << 10);
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_compute: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_move( libxsmm_generated_code*                io_generated_code,
                                           const unsigned int                     i_vmove_instr,
                                           const unsigned char                    i_gp_reg_addr,
                                           const unsigned char                    i_gp_reg_offset,
                                           const short                            i_offset,
                                           const unsigned char                    i_vec_reg,
                                           const unsigned char                    i_pred_reg ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_A64FX ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: at least ARM A64FX needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_vmove_instr ) {
    case LIBXSMM_AARCH64_INSTR_SVE_LDR_P_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_STR_P_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1D_SR:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1W_SR:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1D_SR:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1D_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1W_SR:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF:
       break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: unexpected instruction number: %u\n", i_vmove_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int*)io_generated_code->generated_code;

    /* fix bits */
    code[code_head] = (unsigned int)(0xffffff00 & i_vmove_instr);
    /* setting Rt */
    code[code_head] |= (unsigned int)(0x1f & i_vec_reg);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_addr) << 5);
    /* setting pred register */
    if ( (i_vmove_instr & 0x80) == 0x80 ) {
      code[code_head] |= (unsigned int)((0x7 & i_pred_reg) << 10);
    }

    /* load/store with offset register */
    if ( (i_vmove_instr & 0x3) == 0x3 ) {
      /* setting Rm */
      code[code_head] |= (unsigned int)((0x1f & i_gp_reg_offset) << 16);
    }
    if ( (i_vmove_instr & 0x4) == 0x4 ) {
      if ( (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF) || (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF) ) {
        unsigned char l_offset = 0;
        if ( i_offset < 0 ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: for LD1RW, LD1RD only positive offsets are allowed!\n");
          exit(-1);
        }

        if ( i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF ) {
          l_offset = (unsigned char)((unsigned int)i_offset >> 2);
        } else if ( i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF ) {
          l_offset = (unsigned char)((unsigned int)i_offset >> 3);
        }

        if ( l_offset > 63 ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: for LD1RW, LD1RD offset is out of range!\n");
          exit(-1);
        }

        code[code_head] |= (unsigned int)((0x3f & l_offset) << 16);
      } else if ( (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LDR_P_I_OFF) || (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF) ||
                  (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_STR_P_I_OFF) || (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF)    ) {
        if ( i_offset < -256 || i_offset > 256 ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: for STR/LDR offset is out of range!\n");
          exit(-1);
        }

        code[code_head] |= (unsigned int)((0x7 & i_offset) << 10);
        code[code_head] |= (unsigned int)((0x3f & (i_offset >> 3)) << 16);
      } else {
        if ( i_offset < -8 || i_offset > 7 ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_sve_struct_move: for LD1W/D, ST1W/D offset is out of range!\n");
          exit(-1);
        }

        code[code_head] |= (unsigned int)((0xf & i_offset) << 16);
      }
    }

    /* we have only 16 P registers in the encoding */
    if ( (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LDR_P_I_OFF) || (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_STR_P_I_OFF) ) {
       code[code_head] &= (unsigned int)0xffffffef;
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_struct_move: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_compute( libxsmm_generated_code*        io_generated_code,
                                              const unsigned int             i_vec_instr,
                                              const unsigned char            i_vec_reg_src_0,
                                              const unsigned char            i_vec_reg_src_1,
                                              const unsigned char            i_index,
                                              const unsigned char            i_vec_reg_dst,
                                              const unsigned char            i_pred_reg,
                                              const libxsmm_aarch64_sve_type i_type ) {
  LIBXSMM_UNUSED( i_index );

  if ( io_generated_code->arch < LIBXSMM_AARCH64_A64FX ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_compute: at least ARM A64FX needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_vec_instr ) {
    case LIBXSMM_AARCH64_INSTR_SVE_FMLA_V:
    case LIBXSMM_AARCH64_INSTR_SVE_EOR_V:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_sve_compute: unexpected instruction number: %u\n", i_vec_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* fix bits */
    code[code_head]  = (unsigned int)(0xffffff00 & i_vec_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)(0x1f & i_vec_reg_dst);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_vec_reg_src_0) << 5);
    /* setting Rm */
    code[code_head] |= (unsigned int)((0x1f & i_vec_reg_src_1) << 16);
    if ( i_vec_instr != LIBXSMM_AARCH64_INSTR_SVE_EOR_V ) {
      /* setting type */
      code[code_head] |= (unsigned int)((0x3 & i_type) << 22);
      /* setting p reg */
      code[code_head] |= (unsigned int)((0x7 & i_pred_reg) << 10);
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_compute: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_pcompute( libxsmm_generated_code*           io_generated_code,
                                               const unsigned int                i_pred_instr,
                                               const unsigned char               i_pred_reg,
                                               const unsigned char               i_gp_reg_src_0,
                                               const libxsmm_aarch64_gp_width    i_gp_width,
                                               const unsigned char               i_gp_reg_src_1,
                                               const libxsmm_aarch64_sve_pattern i_pattern,
                                               const libxsmm_aarch64_sve_type    i_type ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_A64FX ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_pcompute: at least ARM A64FX needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_pred_instr ) {
    case LIBXSMM_AARCH64_INSTR_SVE_PTRUE:
      break;
    case LIBXSMM_AARCH64_INSTR_SVE_WHILELT:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_sve_pcompute: unexpected instruction number: %u\n", i_pred_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* fix bits */
    code[code_head]  = (unsigned int)(0xffffff00 & i_pred_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)(0xf & i_pred_reg);
    if( (i_pred_instr & 0x3) == 0x1 ) {
      /* setting pattern */
      code[code_head] |= (unsigned int)((0x1f & i_pattern) << 5);
    }
    else if( (i_pred_instr & 0x3) == 0x3 ) {
      /* setting first source register */
      code[code_head] |= (unsigned int)((0x1f & i_gp_reg_src_0) << 5);
      /* setting width of registers */
      code[code_head] |= (unsigned int)((0x1 & i_gp_width) << 12);
      /* setting second source register */
      code[code_head] |= (unsigned int)((0x1f & i_gp_reg_src_1) << 16);
    }
    /* setting type */
    code[code_head] |= (unsigned int)((0x3 & i_type) << 22);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_pcompute: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_move( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_move_instr,
                                           const unsigned int      i_gp_reg_addr,
                                           const unsigned int      i_gp_reg_offset,
                                           const short             i_offset,
                                           const unsigned char     i_gp_reg_dst ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_move_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_LDR_R:
    case LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF:
    case LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST:
    case LIBXSMM_AARCH64_INSTR_GP_LDR_I_PRE:
    case LIBXSMM_AARCH64_INSTR_GP_STR_R:
    case LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF:
    case LIBXSMM_AARCH64_INSTR_GP_STR_I_POST:
    case LIBXSMM_AARCH64_INSTR_GP_STR_I_PRE:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: unexpected instruction number: %u\n", i_move_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int*)io_generated_code->generated_code;

    /* fix bits */
    code[code_head] = (unsigned int)(0xffffff00 & i_move_instr);
    /* setting Rt */
    code[code_head] |= (unsigned int)(0x1f & i_gp_reg_dst);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_addr) << 5);
   /* setting size */
    code[code_head] |= (unsigned int)((0x20 & i_gp_reg_dst) << 25);

    /* load/store with offset register */
    if ( ((i_move_instr & 0x3) == 0x3) && ((i_move_instr & 0x4) == 0x0) ) {
      /* setting Rm */
      code[code_head] |= (unsigned int)((0x1f & i_gp_reg_offset) << 16);
      /* setting lowest option bit based on the register name */
      code[code_head] |= (unsigned int)((0x20 & i_gp_reg_offset) << 8);
    }

    /* load/store with imm offset */
    if ( (i_move_instr & 0x6) == 0x6 ) {
      /* set imm */
      if ( i_move_instr == LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF ||
           i_move_instr == LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF     ) {
        short l_imm;
        /* adjust offset based on vector width */
        if ( (0x20 & i_gp_reg_dst) == 0x20 ) {
          l_imm = i_offset/8;
        } else {
          l_imm = i_offset/4;
        }
        if ( (l_imm > 0x0fff) || (i_offset < 0) ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: offset for unsigned offnset addressing mode out of range: %i, %i!\n", l_imm, i_offset);
          exit(-1);
        }
        code[code_head] |= (unsigned int)((0x00000fff & l_imm) << 10);
      } else {
        if ( (i_offset < -256) || (i_offset > 255) ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: offset for per-index/post-index addressing mode out of range: %i!\n", i_offset);
          exit(-1);
        }
        code[code_head] |= (unsigned int)((0x000001ff & i_offset) << 12);
      }
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_pair_move( libxsmm_generated_code*           io_generated_code,
                                                const unsigned int                i_move_instr,
                                                const unsigned char               i_gp_reg_addr,
                                                const char                        i_offset,
                                                const unsigned char               i_gp_reg_0,
                                                const unsigned char               i_gp_reg_1 ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_pair_move: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_move_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF:
    case LIBXSMM_AARCH64_INSTR_GP_LDP_I_POST:
    case LIBXSMM_AARCH64_INSTR_GP_LDP_I_PRE:
    case LIBXSMM_AARCH64_INSTR_GP_LDNP_I_OFF:
    case LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF:
    case LIBXSMM_AARCH64_INSTR_GP_STP_I_POST:
    case LIBXSMM_AARCH64_INSTR_GP_STP_I_PRE:
    case LIBXSMM_AARCH64_INSTR_GP_STNP_I_OFF:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_pair_move: unexpected instruction number: %u\n", i_move_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;
    unsigned char l_opc = 0x0;
    signed char l_imm = 0x0;

    if ( (0x20 & i_gp_reg_0) == 0x20 ) {
      l_opc = 0x1;
      l_imm = i_offset/8;
    } else {
      l_opc = 0x0;
      l_imm = i_offset/4;
    }

    if ( (l_imm < -64) || (l_imm > 63) ) {
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: offset out of range: %i!\n", i_offset);
      exit(-1);
    }

    /* fix bits */
    code[code_head] = (unsigned int)(0xffffff00 & i_move_instr);
    /* setting Rt */
    code[code_head] |= (unsigned int)(0x1f & i_gp_reg_0);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_addr) << 5);
    /* setting Rt2 */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_1) << 10);
    /* setting imm7 */
    code[code_head] |= (unsigned int)((0x7f & l_imm) << 15);
    /* set opc */
    code[code_head] |= (unsigned int)(l_opc << 31);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_pair_move: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_move_imm16( libxsmm_generated_code* io_generated_code,
                                                 const unsigned int      i_alu_instr,
                                                 const unsigned int      i_gp_reg_dst,
                                                 const unsigned char     i_shift,
                                                 const unsigned short    i_imm16 ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_move_imm16: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_alu_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_MOVZ:
    case LIBXSMM_AARCH64_INSTR_GP_MOVN:
    case LIBXSMM_AARCH64_INSTR_GP_MOVK:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_move_imm16: unexpected instruction number: %u\n", i_alu_instr);
      exit(-1);
  }

  if ( ((i_gp_reg_dst < LIBXSMM_AARCH64_GP_REG_X0) && (i_shift > 1)) || (i_shift > 3) ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_move_imm16: unexpected shift: %u %u %u\n", i_alu_instr, i_gp_reg_dst, (unsigned int)i_shift);
    exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* computing hw */
    unsigned char l_hw = (unsigned char)( i_gp_reg_dst < LIBXSMM_AARCH64_GP_REG_X0 ) ? (0x1 & i_shift) : (0x3 & i_shift);
    /* fix bits */
    code[code_head]  = (unsigned int)(0xffe00000 & i_alu_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)(0x1f & i_gp_reg_dst);
    /* setting sf */
    code[code_head] |= (unsigned int)((0x20 & i_gp_reg_dst) << 26);
    /* setting imm16 */
    code[code_head] |= (unsigned int)(((unsigned int)i_imm16) << 5);
    /* setting hw */
    code[code_head] |= (unsigned int)(((unsigned int)l_hw) << 21);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_move_imm16: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_set_imm64( libxsmm_generated_code*  io_generated_code,
                                                const unsigned int       i_gp_reg_dst,
                                                const unsigned long long i_imm64 ) {
  if (        i_imm64 <=         0xffff ) {
    libxsmm_aarch64_instruction_alu_move_imm16( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MOVZ,
                                                i_gp_reg_dst, 0, (unsigned short)i_imm64 );
  } else if ( i_imm64 <=     0xffffffff ) {
    libxsmm_aarch64_instruction_alu_move_imm16( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MOVZ,
                                                i_gp_reg_dst, 0, (unsigned short)i_imm64 );
    libxsmm_aarch64_instruction_alu_move_imm16( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MOVK,
                                                i_gp_reg_dst, 1, (unsigned short)( i_imm64 >> 16 ) );
  } else if ( i_imm64 <= 0xffffffffffff ) {
    libxsmm_aarch64_instruction_alu_move_imm16( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MOVZ,
                                                i_gp_reg_dst, 0, (unsigned short)i_imm64 );
    libxsmm_aarch64_instruction_alu_move_imm16( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MOVK,
                                                i_gp_reg_dst, 1, (unsigned short)( i_imm64 >> 16 ) );
    libxsmm_aarch64_instruction_alu_move_imm16( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MOVK,
                                                i_gp_reg_dst, 2, (unsigned short)( i_imm64 >> 32 ) );
  } else {
    libxsmm_aarch64_instruction_alu_move_imm16( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MOVZ,
                                                i_gp_reg_dst, 0, (unsigned short)i_imm64 );
    libxsmm_aarch64_instruction_alu_move_imm16( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MOVK,
                                                i_gp_reg_dst, 1, (unsigned short)( i_imm64 >> 16 ) );
    libxsmm_aarch64_instruction_alu_move_imm16( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MOVK,
                                                i_gp_reg_dst, 2, (unsigned short)( i_imm64 >> 32 ) );
    libxsmm_aarch64_instruction_alu_move_imm16( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_MOVK,
                                                i_gp_reg_dst, 3, (unsigned short)( i_imm64 >> 48 ) );
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_imm12( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_alu_instr,
                                                    const unsigned char     i_gp_reg_src,
                                                    const unsigned char     i_gp_reg_dst,
                                                    const unsigned short    i_imm12,
                                                    const unsigned char     i_imm12_lsl12 ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm12: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_alu_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_ADD_I:
    case LIBXSMM_AARCH64_INSTR_GP_SUB_I:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm12: unexpected instruction number: %u\n", i_alu_instr);
      exit(-1);
  }

  /* check for imm being in range */
  if ( (i_imm12 > 0xfff) || (i_imm12_lsl12 > 1) ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm12: unexpected imm/shift: %u %u %u\n", i_alu_instr, (unsigned int)i_imm12, (unsigned int)i_imm12_lsl12);
    exit(-1);
  }

  /* check that all regs are either 32 or 64 bit */
  if ( ((i_gp_reg_src > 31) && ( i_gp_reg_dst > 31 )) ) {
    /* nothing */
  } else if ( ((i_gp_reg_src < 32) && ( i_gp_reg_dst < 32 )) ) {
    /* nothing */
  } else {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm12: all regsiters need to be either 32 or 64bit; instr: %u\n", i_alu_instr);
    exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* fix bits */
    code[code_head]  = (unsigned int)(0xffc00000 & i_alu_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)(0x1f & i_gp_reg_dst);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_src) << 5);
    /* setting sf */
    code[code_head] |= (unsigned int)((0x20 & i_gp_reg_dst) << 26);
    /* setting imm16 */
    code[code_head] |= (unsigned int)((0xfff & i_imm12) << 10);
    /* setting sh */
    code[code_head] |= (unsigned int)(((unsigned int)i_imm12_lsl12) << 22);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm12: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_imm24( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_alu_instr,
                                                    const unsigned char     i_gp_reg_src,
                                                    const unsigned char     i_gp_reg_dst,
                                                    const unsigned int      i_imm24 ) {
  if ( i_imm24 > 0xffffff ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm24: unexpected imm/shift: %u %u\n", i_alu_instr, i_imm24);
    exit(-1);
  }

  if ( i_imm24 <= 0xfff ) {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, i_alu_instr, i_gp_reg_src, i_gp_reg_dst,
                                                   (unsigned short)(0xfff & i_imm24), 0);
  } else {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, i_alu_instr, i_gp_reg_src, i_gp_reg_dst,
                                                   (unsigned short)(0xfff & i_imm24), 0);
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, i_alu_instr, i_gp_reg_src, i_gp_reg_dst,
                                                   (unsigned short)(0xfff & (i_imm24 >> 12)), 1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_shifted_reg( libxsmm_generated_code*         io_generated_code,
                                                          const unsigned int              i_alu_instr,
                                                          const unsigned char             i_gp_reg_src_0,
                                                          const unsigned char             i_gp_reg_src_1,
                                                          const unsigned char             i_gp_reg_dst,
                                                          const unsigned char             i_imm6,
                                                          const libxsmm_aarch64_shiftmode i_shift_dir ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_shifted_reg: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_alu_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_ADD_SR:
    case LIBXSMM_AARCH64_INSTR_GP_SUB_SR:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_shifted_reg: unexpected instruction number: %u\n", i_alu_instr);
      exit(-1);
  }

  /* check for imm being in range */
  if ( (i_imm6 > 0x3f) ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_shifted_reg: unexpected imm: %u %u\n", i_alu_instr, (unsigned int)i_imm6);
    exit(-1);
  }

  /* check that all regs are either 32 or 64 bit */
  if ( (i_gp_reg_src_0 > 31) && (i_gp_reg_src_1 > 31) && ( i_gp_reg_dst > 31 ) ) {
    /* nothing */
  } else if ( (i_gp_reg_src_0 < 32) && (i_gp_reg_src_1 < 32) && ( i_gp_reg_dst < 32 ) ) {
    /* nothing */
  } else {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_shifted_reg: all regsiters need to be either 32 or 64bit; instr: %u\n", i_alu_instr);
    exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* computing hw */
    unsigned char l_imm = (unsigned char)( i_gp_reg_dst < LIBXSMM_AARCH64_GP_REG_X0 ) ? (0x1f & i_imm6) : (0x3f & i_imm6);
     /* fix bits */
    code[code_head]  = (unsigned int)(0xff000000 & i_alu_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)(0x1f & i_gp_reg_dst);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_src_0) << 5);
    /* setting Rm */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_src_1) << 16);
    /* setting sf */
    code[code_head] |= (unsigned int)((0x20 & i_gp_reg_dst) << 26);
    /* setting imm16 */
    code[code_head] |= (unsigned int)((0x3f & l_imm) << 10);
    /* setting sh */
    code[code_head] |= (unsigned int)((0x3  & i_shift_dir) << 22);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_shifted_reg: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_imm64( libxsmm_generated_code*         io_generated_code,
                                                    const unsigned int              i_alu_meta_instr,
                                                    const unsigned char             i_gp_reg_src,
                                                    const unsigned char             i_gp_reg_tmp,
                                                    const unsigned char             i_gp_reg_dst,
                                                    const unsigned long long        i_imm64 ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm64: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_alu_meta_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_META_ADD:
    case LIBXSMM_AARCH64_INSTR_GP_META_SUB:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm64: unexpected instruction number: %u\n", i_alu_meta_instr);
      exit(-1);
  }

  /* check for imm being in range */
  if ( i_imm64 <= 0xffffff ) {
    unsigned int l_alu_instr;
    unsigned int l_imm24;

    /* map meta insttuction to ISA */
    switch( i_alu_meta_instr ) {
      case LIBXSMM_AARCH64_INSTR_GP_META_ADD:
        l_alu_instr = LIBXSMM_AARCH64_INSTR_GP_ADD_I;
        break;
      case LIBXSMM_AARCH64_INSTR_GP_META_SUB:
        l_alu_instr = LIBXSMM_AARCH64_INSTR_GP_SUB_I;
        break;
      default:
        fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm64: unexpected instruction number (24bit imm): %u\n", i_alu_meta_instr);
        exit(-1);
    }
    l_imm24 = (unsigned int)(i_imm64 & 0xffffff);

    libxsmm_aarch64_instruction_alu_compute_imm24( io_generated_code, l_alu_instr,
                                                   i_gp_reg_src, i_gp_reg_dst, l_imm24 );
  } else {
    unsigned int l_alu_instr;

    /* map meta insttuction to ISA */
    switch( i_alu_meta_instr ) {
      case LIBXSMM_AARCH64_INSTR_GP_META_ADD:
        l_alu_instr = LIBXSMM_AARCH64_INSTR_GP_ADD_SR;
        break;
      case LIBXSMM_AARCH64_INSTR_GP_META_SUB:
        l_alu_instr = LIBXSMM_AARCH64_INSTR_GP_SUB_SR;
        break;
      default:
        fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm64: unexpected instruction number (64bit imm): %u\n", i_alu_meta_instr);
        exit(-1);
    }

    /* move imm64 into the temp register */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp, i_imm64 );

    /* reg-reg instruction */
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, l_alu_instr,
                                                         i_gp_reg_src, i_gp_reg_tmp, i_gp_reg_dst,
                                                         0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_register_jump_back_label( libxsmm_generated_code*     io_generated_code,
                                                           libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_register_jump_back_label: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  /* check if we still have label we can jump to */
  if ( io_loop_label_tracker->label_count == 512 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    int l_lab = io_loop_label_tracker->label_count;
    io_loop_label_tracker->label_count++;
    io_loop_label_tracker->label_address[l_lab] = io_generated_code->code_size;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_register_jump_back_label: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_cond_jump_back_to_label( libxsmm_generated_code*     io_generated_code,
                                                          const unsigned int          i_jmp_instr,
                                                          const unsigned int          i_gp_reg_cmp,
                                                          libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_cond_jump_back_to_label: at least ARM V81 needs to be specified as target arch!\n");
    exit(-1);
  }

  switch ( i_jmp_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_CBNZ:
    case LIBXSMM_AARCH64_INSTR_GP_CBZ:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_cond_jump_back_to_label: unexpected instruction number: %u\n", i_jmp_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_lab = --io_loop_label_tracker->label_count;
    unsigned int l_jmp_dst = (io_loop_label_tracker->label_address[l_lab])/4;
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* computing jump immediate */
    int l_jmp_imm = (int)l_jmp_dst - (int)code_head;
     /* fix bits */
    code[code_head]  = (unsigned int)(0xff000000 & i_jmp_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)(0x1f & i_gp_reg_cmp);
    /* setting sf */
    code[code_head] |= (unsigned int)((0x20 & i_gp_reg_cmp) << 26);
     /* setting imm16 */
    code[code_head] |= (unsigned int)((0x7ffff & l_jmp_imm) << 5);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_cond_jump_back_to_label: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

#if 0
LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_register_jump_label( libxsmm_generated_code*          io_generated_code,
                                                      const unsigned int          i_label_no,
                                                      libxsmm_jump_label_tracker* io_jump_label_tracker ) {
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_jump_to_label( libxsmm_generated_code*     io_generated_code,
                                                const unsigned int            i_jmp_instr,
                                                const unsigned int            i_label_no,
                                                libxsmm_jump_label_tracker* io_jump_label_tracker ) {
}
#endif

