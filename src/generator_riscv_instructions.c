/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer, Antonio Noack (FSU Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_riscv_instructions.h"

#define FILL_REGID(r, t) (t & (r << libxsmm_ctz(t)));

/* RVV LD/ST */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_rvv_move( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_vmove_instr,
                                           const unsigned int      i_gp_reg_addr,
                                           const unsigned int      i_reg_offset_idx,
                                           const int               i_offset,
                                           const unsigned int      i_vec_reg,
                                           const unsigned int      i_pred_reg ) {
  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_riscv_instruction_rvv_move: at least RISCV needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RVV compute instruction */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_rvv_compute( libxsmm_generated_code*  io_generated_code,
                                              const unsigned int       i_vec_instr,
                                              const unsigned int       i_vec_reg_src_0,
                                              const unsigned int       i_vec_reg_src_1,
                                              const unsigned char      i_index,
                                              const unsigned int       i_vec_reg_dst,
                                              const unsigned int       i_pred_reg,
                                              const libxsmm_riscv_type i_type ) {
}

/* RVV predicted compute */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_rvv_pcompute( libxsmm_generated_code*      io_generated_code,
                                               const unsigned int           i_pred_instr,
                                               const unsigned int           i_pred_reg,
                                               const unsigned int           i_gp_reg_src_0,
                                               const libxsmm_riscv_gp_width i_gp_width,
                                               const unsigned int           i_gp_reg_src_1,
                                               const libxsmm_riscv_pattern  i_pattern,
                                               const libxsmm_riscv_type     i_type ) {
}

/* RV64 base ISA LD/ST */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_alu_move( libxsmm_generated_code* io_generated_code,
                                         const unsigned int       i_move_instr,
                                         const unsigned int       i_gp_reg_addr,
                                         const int                i_offset,
                                         const unsigned int       i_gp_reg_dst ) {
  int is_load = 0;

  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_riscv_instruction_move: at least RISCV needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_move_instr ) {
    case LIBXSMM_RISCV_INSTR_GP_LB:
    case LIBXSMM_RISCV_INSTR_GP_LH:
    case LIBXSMM_RISCV_INSTR_GP_LW:
    case LIBXSMM_RISCV_INSTR_GP_LD:
    case LIBXSMM_RISCV_INSTR_GP_LQ:
      is_load = 1;
      break;
    case LIBXSMM_RISCV_INSTR_GP_SB:
    case LIBXSMM_RISCV_INSTR_GP_SH:
    case LIBXSMM_RISCV_INSTR_GP_SW:
    case LIBXSMM_RISCV_INSTR_GP_SD:
    case LIBXSMM_RISCV_INSTR_GP_SQ:
      break;
    default:
      fprintf(stderr, "libxsmm_riscv_instruction_move: unexpected instruction number: %u\n", i_move_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int*)io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* fix bits */
    code[code_head]  = i_move_instr;

    if (is_load) {
      /* setting RD */
      code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RISCV_INSTR_FIELD_RD);
    } else {
      /* setting RS2 */
      code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RISCV_INSTR_FIELD_RS2);
    }

    /* setting RS1 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_addr, LIBXSMM_RISCV_INSTR_FIELD_RS1);

    /* setting IMM */
    code[code_head] |= (unsigned int)FILL_REGID(i_offset, LIBXSMM_RISCV_INSTR_FIELD_IMM12);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_riscv_instruction_move: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 base ISA R-type instructions */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_alu_compute( libxsmm_generated_code* io_generated_code,
                                            const unsigned int      i_alu_instr,
                                            const unsigned int      i_gp_reg_src_1,
                                            const unsigned int      i_gp_reg_src_2,
                                            const unsigned int      i_gp_reg_dst) {
  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm12: at least RISCV needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  // Sanity check
  switch ( i_alu_instr ) {
    case LIBXSMM_RISCV_INSTR_GP_ADD:
    case LIBXSMM_RISCV_INSTR_GP_SUB:
    case LIBXSMM_RISCV_INSTR_GP_OR:
    case LIBXSMM_RISCV_INSTR_GP_AND:
    case LIBXSMM_RISCV_INSTR_GP_XOR:
    case LIBXSMM_RISCV_INSTR_GP_SRL:
    case LIBXSMM_RISCV_INSTR_GP_SLL:
    case LIBXSMM_RISCV_INSTR_GP_SLT:
      break;
    default:
      fprintf(stderr, "libxsmm_riscv_instruction_alu_compute: unexpected instruction number: %u\n", i_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  /* check that all regs are either 32 or 64 bit */
  if ( ((i_gp_reg_src_1 > 31) && (i_gp_reg_src_2 > 31) && ( i_gp_reg_dst > 31 )) ) {
    /* nothing */
  } else if ( ((i_gp_reg_src_1 < 32) && (i_gp_reg_src_2 < 32) && ( i_gp_reg_dst < 32 )) ) {
    /* nothing */
  } else {
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm12: all regsiters need to be either 32 or 64bit; instr: %u\n", i_alu_instr);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /*
     *  i_alu_instr
     *
     * */
    /* fix bits */
    code[code_head]  = i_alu_instr;
    /* setting RD */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_1, LIBXSMM_RISCV_INSTR_FIELD_RS1);
    /* setting RS1 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_2, LIBXSMM_RISCV_INSTR_FIELD_RS2);
    /* setting RS2 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RISCV_INSTR_FIELD_RD);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 base ISA I-type instructions */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_alu_compute_imm12( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_alu_instr,
                                                    const unsigned int      i_gp_reg_src,
                                                    const unsigned int      i_gp_reg_dst,
                                                    const unsigned char     i_imm12 ) {
  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm12: at least RISCV needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  // Sanity check
  switch ( i_alu_instr ) {
    case LIBXSMM_RISCV_INSTR_GP_ADD:
    case LIBXSMM_RISCV_INSTR_GP_SUB:
    case LIBXSMM_RISCV_INSTR_GP_OR:
    case LIBXSMM_RISCV_INSTR_GP_AND:
    case LIBXSMM_RISCV_INSTR_GP_XOR:
    case LIBXSMM_RISCV_INSTR_GP_SRL:
    case LIBXSMM_RISCV_INSTR_GP_SLL:
    case LIBXSMM_RISCV_INSTR_GP_SLT:
      break;
    default:
      fprintf(stderr, "libxsmm_riscv_instruction_alu_compute: unexpected instruction number: %u\n", i_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  /* check that all regs are either 32 or 64 bit */
  if ( ((i_gp_reg_src > 31) && ( i_gp_reg_dst > 31 )) ) {
    /* nothing */
  } else if ( ((i_gp_reg_src < 32) && ( i_gp_reg_dst < 32 )) ) {
    /* nothing */
  } else {
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm12: all regsiters need to be either 32 or 64bit; instr: %u\n", i_alu_instr);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /*
     *  i_alu_instr
     * */
    /* fix bits */
    code[code_head]  = i_alu_instr;
    /* setting RD */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RISCV_INSTR_FIELD_RD);
    /* setting RS1 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src, LIBXSMM_RISCV_INSTR_FIELD_RS1);
    /* setting IMM12 */
    code[code_head] |= (unsigned int)FILL_REGID(i_imm12, LIBXSMM_RISCV_INSTR_FIELD_IMM12);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm12: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 base ISA U-type instructions */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_alu_compute_imm20( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_alu_instr,
                                                    const unsigned int      i_gp_reg_dst,
                                                    const unsigned int      i_imm20 ) {
  if ( i_imm20 > 0xfffff ) {
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm20: unexpected imm: %u %u\n", i_alu_instr, i_imm20);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

  /* fix bits */
  code[code_head]  = i_alu_instr;
  /* setting RD */
  code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RISCV_INSTR_FIELD_RD);
  /* setting RS1 */
  code[code_head] |= (unsigned int)FILL_REGID(i_imm20, LIBXSMM_RISCV_INSTR_FIELD_IMM20);

  /* advance code head */
  io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm12: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 base ISA U-type instructions */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_alu_move_imm12( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_alu_instr,
                                               const unsigned int      i_gp_reg_dst,
                                               const unsigned int      i_imm12 ) {
  // ADDI immediate to X0 register
  libxsmm_riscv_instruction_alu_compute_imm12(io_generated_code,
    LIBXSMM_RISCV_INSTR_GP_ADDI, LIBXSMM_RISCV_GP_REG_X0, i_gp_reg_dst, i_imm12);
}

/* 64 bit immediate move using addi, lui, and shift instructions. */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_alu_set_imm64( libxsmm_generated_code*  io_generated_code,
                                              const unsigned int       i_gp_reg_dst,
                                              const unsigned long long i_imm64 ) {
#define IMM_12_1 (0xfff)
#define IMM_20_1 (0xfffff000)
#define IMM_12_2 (0xfff00000000)
#define IMM_20_2 (0xfffff00000000000)

  unsigned int imm_12_1 = (i_imm64 & IMM_12_1);
  unsigned int imm_20_1 = ((i_imm64 & IMM_20_1) >> 12);
  unsigned int imm_12_2 = ((i_imm64 & IMM_12_2) >> 32);
  unsigned int imm_20_2 = ((i_imm64 & IMM_20_2) >> 44);

  // LUI 20 bits
  libxsmm_riscv_instruction_alu_compute_imm20(io_generated_code,
      LIBXSMM_RISCV_INSTR_GP_LUI, i_gp_reg_dst, imm_20_2);
  // ADD 16 bits
  libxsmm_riscv_instruction_alu_compute_imm12(io_generated_code,
      LIBXSMM_RISCV_INSTR_GP_ADDI, i_gp_reg_dst, i_gp_reg_dst, imm_12_2);
  // SHIFT Left
  libxsmm_riscv_instruction_alu_compute_imm12(io_generated_code,
      LIBXSMM_RISCV_INSTR_GP_SLLI, i_gp_reg_dst, i_gp_reg_dst, 32);
  // LUI 20 bits
  libxsmm_riscv_instruction_alu_compute_imm20(io_generated_code,
      LIBXSMM_RISCV_INSTR_GP_LUI, i_gp_reg_dst, imm_20_1);
  // ADD 16 bits
  libxsmm_riscv_instruction_alu_compute_imm12(io_generated_code,
      LIBXSMM_RISCV_INSTR_GP_ADDI, i_gp_reg_dst, i_gp_reg_dst, imm_12_1);

#undef IMM_12_1
#undef IMM_20_1
#undef IMM_12_2
#undef IMM_20_2
}

/* 64-bit compute with immediate uses 64-bit move and alu instructions. */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_alu_compute_imm64( libxsmm_generated_code*  io_generated_code,
                                                  const unsigned int       i_alu_meta_instr,
                                                  const unsigned int       i_gp_reg_src,
                                                  const unsigned int       i_gp_reg_tmp,
                                                  const unsigned int       i_gp_reg_dst,
                                                  const unsigned long long i_imm64 ) {
  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm64: at least RISCV needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* move imm64 into the temp register */
  libxsmm_riscv_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp, i_imm64 );

  /* reg-reg instruction */
  libxsmm_riscv_instruction_alu_compute( io_generated_code, i_alu_meta_instr,
                                         i_gp_reg_src, i_gp_reg_tmp, i_gp_reg_dst);
}

/* Conditional jump instructions. */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_cond_jump( libxsmm_generated_code* io_generated_code,
                                          const unsigned int      i_jmp_instr,
                                          const unsigned int      i_gp_reg_src_1,
                                          const unsigned int      i_gp_reg_src_2,
                                          const unsigned int      i_imm ) {
  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_riscv_instruction_cond_jump_back_to_label: at least RISCV needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

  /* Generate immediate */
  unsigned int imm_lo = ((i_imm >> 11) & 0x1) | (i_imm & 0x1e);
  unsigned int imm_hi = (i_imm & 0x1fe0) | ((i_imm >> 12) & 0x1);

  /* fix bits */
  code[code_head]  = i_jmp_instr;
  /* setting RS1 */
  code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_1, LIBXSMM_RISCV_INSTR_FIELD_RS1);
  /* setting RS2 */
  code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_2, LIBXSMM_RISCV_INSTR_FIELD_RS2);
  /* setting IMM12HI */
  code[code_head] |= (unsigned int)FILL_REGID(imm_hi, LIBXSMM_RISCV_INSTR_FIELD_BIMM12HI);
  /* setting IMM12LO */
  code[code_head] |= (unsigned int)FILL_REGID(imm_lo, LIBXSMM_RISCV_INSTR_FIELD_BIMM12LO);

  /* advance code head */
  io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm12: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 Base UJ-type instructions. */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_jump_and_link( libxsmm_generated_code* io_generated_code,
                                     const unsigned int      i_jmp_instr,
                                     const unsigned int      i_gp_reg_dst,
                                     const unsigned int      i_imm ) {
  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_riscv_instruction_cond_jump_back_to_label: at least RISCV needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

  /* Generate immediate */
  unsigned int imm_lo = (((i_imm & 0x7ff) >> 2)|((i_imm >> 12) & 0xff));
  unsigned int imm_hi = (i_imm & 0x7fe) | ((i_imm & 0x100000) >> 9);
  unsigned int imm_f  = imm_lo | (imm_hi << 8);

  /* fix bits */
  code[code_head]  = i_jmp_instr;
  /* setting RS1 */
  code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RISCV_INSTR_FIELD_RD);
  /* setting IMM20 */
  code[code_head] |= (unsigned int)FILL_REGID(imm_f, LIBXSMM_RISCV_INSTR_FIELD_IMM12HI);

  /* advance code head */
  io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm12: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 Base uncontional jump instructions. */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_jump_and_link_reg( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_jmp_instr,
                                              const unsigned int      i_gp_reg_dst,
                                              const unsigned int      i_gp_reg_src_1,
                                              const unsigned int      i_imm12 ) {
  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_riscv_instruction_cond_jump_back_to_label: at least RISCV needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

  /* fix bits */
  code[code_head]  = i_jmp_instr;
  /* setting RS1 */
  code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RISCV_INSTR_FIELD_RD);
  /* setting RS2 */
  code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_1, LIBXSMM_RISCV_INSTR_FIELD_RS1);
  /* setting IMM12HI */
  code[code_head] |= (unsigned int)FILL_REGID(i_imm12, LIBXSMM_RISCV_INSTR_FIELD_IMM12);

  /* advance code head */
  io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_riscv_instruction_alu_compute_imm12: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* Required : same as ARM64 */
LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_register_jump_label( libxsmm_generated_code*     io_generated_code,
                                                    const unsigned int          i_label_no,
                                                    libxsmm_jump_label_tracker* io_jump_label_tracker ) {
  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_riscv_instruction_register_jump_label: at least RISCV needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* check if the label we try to set is still available */
  if ( io_jump_label_tracker->label_address[i_label_no] > 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_JMPLBL_USED );
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_ref = 0;
    libxsmm_jump_source l_source = io_jump_label_tracker->label_source[i_label_no];
    unsigned int* code = (unsigned int *)io_generated_code->generated_code;

    /* first added label to tracker */
    io_jump_label_tracker->label_address[i_label_no] = io_generated_code->code_size;

    /* patching all previous references */
    for ( l_ref = 0; l_ref < l_source.ref_count; ++l_ref ) {
      int l_distance = (int)io_jump_label_tracker->label_address[i_label_no] - (int)l_source.instr_addr[l_ref];
      l_distance /= 4;

      code[l_source.instr_addr[l_ref]/4] |= (unsigned int)((0x7ffff & l_distance) << 5);
    }
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_register_jump_back_label: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_cond_jump_to_label( libxsmm_generated_code*     io_generated_code,
                                                   const unsigned int          i_jmp_instr,
                                                   const unsigned int          i_gp_reg_cmp,
                                                   const unsigned int          i_label_no,
                                                   libxsmm_jump_label_tracker* io_jump_label_tracker ) {
  unsigned int l_pos;

  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_riscv_instruction_cond_jump_to_label: at least RISCV needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
  /* check if the label we are trying to set is in bounds */
  if ( 512 <= i_label_no ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  /* check if we still have a label we can jump to */
  if ( io_jump_label_tracker->label_source[i_label_no].ref_count == 512-1 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  switch ( i_jmp_instr ) {
    case LIBXSMM_RISCV_INSTR_GP_BEQ:
    case LIBXSMM_RISCV_INSTR_GP_BGE:
    case LIBXSMM_RISCV_INSTR_GP_BGEU:
    case LIBXSMM_RISCV_INSTR_GP_BLT:
    case LIBXSMM_RISCV_INSTR_GP_BLTU:
    case LIBXSMM_RISCV_INSTR_GP_BNE:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_cond_jump_back_to_label: unexpected instruction number: %u\n", i_jmp_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  /* add addr at current position and instruction to tracking structure */
  l_pos = io_jump_label_tracker->label_source[i_label_no].ref_count;
  io_jump_label_tracker->label_source[i_label_no].instr_type[l_pos] = i_jmp_instr;
  io_jump_label_tracker->label_source[i_label_no].instr_addr[l_pos] = io_generated_code->code_size;
  io_jump_label_tracker->label_source[i_label_no].ref_count++;

  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_jmp_dst = (io_jump_label_tracker->label_address[i_label_no]) / 4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;
    unsigned int code_head = io_generated_code->code_size / 4;
    int l_jmp_imm = (l_jmp_dst == 0) /* computing jump immediate */
      ? 0 : (int)l_jmp_dst - (int)code_head;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    // TODO: comd jump
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
    fprintf(stderr, "libxsmm_aarch64_instruction_cond_jump_to_label: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_register_jump_back_label( libxsmm_generated_code*     io_generated_code,
                                                           libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_register_jump_back_label: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* check if we still have a label we can jump to */
  if ( 512 <= io_loop_label_tracker->label_count ) {
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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_riscv_instruction_cond_jump_back_to_label( libxsmm_generated_code*     io_generated_code,
                                                          const unsigned int          i_jmp_instr,
                                                          const unsigned int          i_gp_reg_cmp,
                                                          libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  if ( io_generated_code->arch < LIBXSMM_RISCV ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_cond_jump_back_to_label: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_jmp_instr ) {
    case LIBXSMM_RISCV_INSTR_GP_BEQ:
    case LIBXSMM_RISCV_INSTR_GP_BGE:
    case LIBXSMM_RISCV_INSTR_GP_BGEU:
    case LIBXSMM_RISCV_INSTR_GP_BLT:
    case LIBXSMM_RISCV_INSTR_GP_BLTU:
    case LIBXSMM_RISCV_INSTR_GP_BNE:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_cond_jump_back_to_label: unexpected instruction number: %u\n", i_jmp_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int* code = (unsigned int*)io_generated_code->generated_code;
    unsigned int l_lab = --io_loop_label_tracker->label_count;
    unsigned int l_jmp_dst = (io_loop_label_tracker->label_address[l_lab]) / 4;
    unsigned int code_head = io_generated_code->code_size / 4;
    int l_jmp_imm = (int)l_jmp_dst - (int)code_head; /* computing jump immediate */

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

#undef FILL_REGID
