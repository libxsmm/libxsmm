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

#include "generator_rv64_instructions.h"

#define FILL_REGID(r, t)        (t & (r << libxsmm_ctz(t)));
#define REG_VALID_1(r1)         (((r1) <= LIBXSMM_RV64_GP_REG_X31))
#define REG_VALID_2(r1, r2)     (REG_VALID_1(r1) && REG_VALID_1(r2))
#define REG_VALID_3(r1, r2, r3) (REG_VALID_2(r1, r2) && REG_VALID_1(r3))

LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_open_stream( libxsmm_generated_code* io_generated_code,
                                              const unsigned short    i_callee_save_bitmask )
{
  return;
}

LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_close_stream( libxsmm_generated_code* io_generated_code,
                                               const unsigned short    i_callee_save_bitmask )
{
  libxsmm_rv64_instruction_jump_and_link_reg(io_generated_code, LIBXSMM_RV64_INSTR_GP_JALR, LIBXSMM_RV64_GP_REG_X0, LIBXSMM_RV64_GP_REG_X1, 0);
  return;
}

/* RVV VL config */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_rvv_setvli( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_gp_reg_src,
                                           const unsigned int      i_reg_dst,
                                           const unsigned int      i_sew,
                                           const unsigned int      i_lmul ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setvli: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_2(i_gp_reg_src, i_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setvli: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_sew > 0x8 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setvli: unexpected imm: %u \n", i_sew);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_lmul > 0x8 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setvli: unexpected imm: %u \n", i_lmul);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;
    unsigned int zimm11    = (i_sew << 3 | i_lmul) & 0x3f;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }


    /* fix bits */
    code[code_head]  = LIBXSMM_RV64_INSTR_GP_VSETVLI;

    /* setting RS1 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src, LIBXSMM_RV64_INSTR_FIELD_RS1);

    /* setting RD */
    code[code_head] |= (unsigned int)FILL_REGID(i_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);

    /* setting lmul and sew */
    code[code_head] |= (unsigned int)FILL_REGID(zimm11, LIBXSMM_RV64_INSTR_FIELD_ZIMM11);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setvli: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_rvv_setivli( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_rvl,
                                           const unsigned int      i_reg_dst,
                                           const unsigned int      i_sew,
                                           const unsigned int      i_lmul ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setivli: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_1(i_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setivli: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_sew > 0x8 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setivli: unexpected imm: %u \n", i_sew);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_lmul > 0x8 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setivli: unexpected imm: %u \n", i_lmul);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_rvl > 0x1f ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setivli: unexpected imm: %u \n", i_lmul);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;
    unsigned int uimm      = (i_sew << 3 | i_lmul) & 0x3f;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* fix bits */
    code[code_head]  = LIBXSMM_RV64_INSTR_GP_VSETIVLI;

    /* setting RVL */
    code[code_head] |= (unsigned int)FILL_REGID(i_rvl, LIBXSMM_RV64_INSTR_FIELD_SIMM5);

    /* setting RD */
    code[code_head] |= (unsigned int)FILL_REGID(i_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);

    /* setting lmul and sew */
    code[code_head] |= (unsigned int)FILL_REGID(uimm, LIBXSMM_RV64_INSTR_FIELD_ZIMM11);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setivli: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_rvv_setvl( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_gp_reg_src_1,
                                           const unsigned int      i_gp_reg_src_2,
                                           const unsigned int      i_reg_dst ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setvl: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_3(i_gp_reg_src_1, i_gp_reg_src_2, i_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setvl: invalid register!\n");
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
    code[code_head]  = LIBXSMM_RV64_INSTR_GP_VSETVL;

    /* setting RS1 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_1, LIBXSMM_RV64_INSTR_FIELD_RS1);

    /* setting RD */
    code[code_head] |= (unsigned int)FILL_REGID(i_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);

    /* setting RS2 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_2, LIBXSMM_RV64_INSTR_FIELD_RS2);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_setvl: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RVV LD/ST */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_rvv_move( libxsmm_generated_code* io_generated_code,
                                         const unsigned int      i_vmove_instr,
                                         const unsigned int      i_vec_reg_addr,
                                         const unsigned int      i_vec_reg_offset,
                                         const unsigned int      i_vec_reg_dst,
                                         const unsigned int      i_masked) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_move: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

#define RVV_M(i)    ((i == RVI(VLM_V)) || (i == RVI(VSM_V)))

#define RVV_LU(i)   ((i == RVI(VLE8_V))||(i == RVI(VLE16_V))||(i == RVI(VLE32_V))||(i == RVI(VLE64_V)))
#define RVV_SU(i)   ((i == RVI(VSE8_V))||(i == RVI(VSE16_V))||(i == RVI(VSE32_V))||(i == RVI(VSE64_V)))
#define RVV_U(i)    (RVV_LU(i) || RVV_SU(i))

#define RVV_LS(i)   ((i == RVI(VLSE8_V))||(i == RVI(VLSE16_V))||(i == RVI(VLSE32_V))||(i == RVI(VLSE64_V)))
#define RVV_SS(i)   ((i == RVI(VSSE8_V))||(i == RVI(VSSE16_V))||(i == RVI(VSSE32_V))||(i == RVI(VSSE64_V)))
#define RVV_S(i)    (RVV_LS(i) || RVV_SS(i))

#define RVV_LUI(i)  ((i == RVI(VLUXEI8_V))||(i == RVI(VLUXEI16_V))||(i == RVI(VLUXEI32_V))||(i == RVI(VLUXEI64_V)))
#define RVV_LOI(i)  ((i == RVI(VLOXEI8_V))||(i == RVI(VLOXEI16_V))||(i == RVI(VLOXEI32_V))||(i == RVI(VLOXEI64_V)))
#define RVV_SUI(i)  ((i == RVI(VSUXEI8_V))||(i == RVI(VSUXEI16_V))||(i == RVI(VSUXEI32_V))||(i == RVI(VSUXEI64_V)))
#define RVV_SOI(i)  ((i == RVI(VSOXEI8_V))||(i == RVI(VSOXEI16_V))||(i == RVI(VSOXEI32_V))||(i == RVI(VSOXEI64_V)))
#define RVV_I(i)    (RVV_LUI(i) || RVV_LOI(i) || RVV_SUI(i) || RVV_SOI(i))

  /* Unit stride and mask memory ops */
  if ( RVV_U(i_vmove_instr)||(RVV_M(i_vmove_instr)) ) {
    if ( !REG_VALID_2(i_vec_reg_addr, i_vec_reg_dst) ) {
      fprintf(stderr, "libxsmm_rv64_instruction_rvv_move: invalid register!\n");
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
      code[code_head]  = i_vmove_instr;

      /* setting RS1 */
      code[code_head] |= (unsigned int)FILL_REGID(i_vec_reg_addr, LIBXSMM_RV64_INSTR_FIELD_RS1);

      /* setting RD */
      code[code_head] |= (unsigned int)FILL_REGID(i_vec_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);

      /* setting mask bit */
      code[code_head] |= (unsigned int)FILL_REGID(i_masked, LIBXSMM_RV64_INSTR_FIELD_VM);

      /* advance code head */
      io_generated_code->code_size += 4;
    } else {
      /* assembly not supported right now */
      fprintf(stderr, "libxsmm_rv64_instruction_rvv_move: inline/pure assembly print is not supported!\n");
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }
  }

  /* Stride and Indexed memory ops */
  if ( RVV_S(i_vmove_instr) || RVV_I(i_vmove_instr)) {
    if ( !REG_VALID_3(i_vec_reg_addr, i_vec_reg_offset, i_vec_reg_dst) ) {
      fprintf(stderr, "libxsmm_rv64_instruction_rvv_move: invalid register!\n");
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
      code[code_head]  = i_vmove_instr;

      /* setting RS1 */
      code[code_head] |= (unsigned int)FILL_REGID(i_vec_reg_addr, LIBXSMM_RV64_INSTR_FIELD_RS1);

      /* setting RD */
      code[code_head] |= (unsigned int)FILL_REGID(i_vec_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);

      /* setting RS2 */
      code[code_head] |= (unsigned int)FILL_REGID(i_vec_reg_offset, LIBXSMM_RV64_INSTR_FIELD_RS2);

      /* setting mask bit */
      code[code_head] |= (unsigned int)FILL_REGID(i_masked, LIBXSMM_RV64_INSTR_FIELD_VM);

      /* advance code head */
      io_generated_code->code_size += 4;
    } else {
      /* assembly not supported right now */
      fprintf(stderr, "libxsmm_rv64_instruction_rvv_move: inline/pure assembly print is not supported!\n");
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }
  }

#undef RVV_LU
#undef RVV_SU
#undef RVV_U
#undef RVV_M
#undef RVV_LS
#undef RVV_SS
#undef RVV_S
#undef RVV_LUI
#undef RVV_LOI
#undef RVV_SUI
#undef RVV_SOI
#undef RVV_I
}

/* RVV compute instruction */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_rvv_compute( libxsmm_generated_code*  io_generated_code,
                                            const unsigned int     i_vec_instr,
                                            const unsigned int     i_vec_reg_src_1,
                                            const unsigned int     i_vec_reg_src_2,
                                            const unsigned int     i_vec_reg_dst,
                                            const unsigned int     i_masked) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_compute: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_3(i_vec_reg_src_1, i_vec_reg_src_2, i_vec_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_compute: invalid register!\n");
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
    code[code_head]  = i_vec_instr;

    /* setting RS1 */
    code[code_head] |= (unsigned int)FILL_REGID(i_vec_reg_src_1, LIBXSMM_RV64_INSTR_FIELD_RS1);

    /* setting RD */
    code[code_head] |= (unsigned int)FILL_REGID(i_vec_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);

    /* setting RS2 */
    code[code_head] |= (unsigned int)FILL_REGID(i_vec_reg_src_2, LIBXSMM_RV64_INSTR_FIELD_RS2);

    /* setting mask bit */
    code[code_head] |= (unsigned int)FILL_REGID(i_masked, LIBXSMM_RV64_INSTR_FIELD_VM);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_compute: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_rvv_compute_imm( libxsmm_generated_code*  io_generated_code,
                                              const unsigned int     i_vec_instr,
                                              const unsigned int     i_vec_reg_src,
                                              const unsigned int     i_imm,
                                              const unsigned int     i_reg_dst,
                                              const unsigned int     i_masked) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_compute_imm: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_2(i_vec_reg_src, i_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_compute_imm: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_imm > 0x1f ) {
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_compute_compute_imm: unexpected imm: %u %u\n", i_vec_instr, i_imm);
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
    code[code_head]  = i_vec_instr;

    /* setting RS1 */
    code[code_head] |= (unsigned int)FILL_REGID(i_vec_reg_src, LIBXSMM_RV64_INSTR_FIELD_RS1);

    /* setting RD */
    code[code_head] |= (unsigned int)FILL_REGID(i_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);

    /* setting IMM */
    code[code_head] |= (unsigned int)FILL_REGID(i_imm, LIBXSMM_RV64_INSTR_FIELD_SIMM5);

    /* setting mask bit */
    code[code_head] |= (unsigned int)FILL_REGID(i_masked, LIBXSMM_RV64_INSTR_FIELD_VM);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_rvv_compute_imm: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 base ISA LD/ST */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_alu_move( libxsmm_generated_code* io_generated_code,
                                         const unsigned int       i_move_instr,
                                         const unsigned int       i_gp_reg_addr,
                                         const unsigned int       i_gp_reg_dst,
                                         const int                i_offset ) {
  int is_load = 0;

  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_move: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_2(i_gp_reg_addr, i_gp_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_move: invalid register id !\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_offset > 0xfff ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_move: unexpected imm: %u %u\n",
      i_move_instr, i_offset);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_move_instr ) {
    case LIBXSMM_RV64_INSTR_GP_LB:
    case LIBXSMM_RV64_INSTR_GP_LH:
    case LIBXSMM_RV64_INSTR_GP_LW:
    case LIBXSMM_RV64_INSTR_GP_LD:
    case LIBXSMM_RV64_INSTR_GP_LQ:
      is_load = 1;
      break;
    case LIBXSMM_RV64_INSTR_GP_SB:
    case LIBXSMM_RV64_INSTR_GP_SH:
    case LIBXSMM_RV64_INSTR_GP_SW:
    case LIBXSMM_RV64_INSTR_GP_SD:
    case LIBXSMM_RV64_INSTR_GP_SQ:
      break;
    default:
      fprintf(stderr, "libxsmm_rv64_instruction_alu_move: unexpected instruction number: %u\n", i_move_instr);
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
      code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);

      /* setting IMM */
      code[code_head] |= (unsigned int)FILL_REGID(i_offset, LIBXSMM_RV64_INSTR_FIELD_IMM12);
    } else {
      /* setting RS2 */
      code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RS2);

      /* setting IMM */
      code[code_head] |= (unsigned int)FILL_REGID((i_offset & 0x1f), LIBXSMM_RV64_INSTR_FIELD_IMM12LO);
      code[code_head] |= (unsigned int)FILL_REGID((i_offset >> 5), LIBXSMM_RV64_INSTR_FIELD_IMM12HI);
    }

    /* setting RS1 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_addr, LIBXSMM_RV64_INSTR_FIELD_RS1);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_alu_move: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 base ISA R-type instructions */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_alu_compute( libxsmm_generated_code* io_generated_code,
                                            const unsigned int      i_alu_instr,
                                            const unsigned int      i_gp_reg_src_1,
                                            const unsigned int      i_gp_reg_src_2,
                                            const unsigned int      i_gp_reg_dst) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_3(i_gp_reg_src_1, i_gp_reg_src_2, i_gp_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* Sanity check */
  switch ( i_alu_instr ) {
    case LIBXSMM_RV64_INSTR_GP_ADD:
    case LIBXSMM_RV64_INSTR_GP_SUB:
    case LIBXSMM_RV64_INSTR_GP_OR:
    case LIBXSMM_RV64_INSTR_GP_AND:
    case LIBXSMM_RV64_INSTR_GP_XOR:
    case LIBXSMM_RV64_INSTR_GP_SRL:
    case LIBXSMM_RV64_INSTR_GP_SLL:
    case LIBXSMM_RV64_INSTR_GP_SLLW:
    case LIBXSMM_RV64_INSTR_GP_SLT:
    case LIBXSMM_RV64_INSTR_GP_SLTU:
    case LIBXSMM_RV64_INSTR_GP_SRA:
    case LIBXSMM_RV64_INSTR_GP_SRAW:
    case LIBXSMM_RV64_INSTR_GP_SRLW:
    case LIBXSMM_RV64_INSTR_GP_SUBW:
    case LIBXSMM_RV64_INSTR_GP_MUL:
    case LIBXSMM_RV64_INSTR_GP_MULW:
    case LIBXSMM_RV64_INSTR_GP_MULH:
    case LIBXSMM_RV64_INSTR_GP_MULHU:
    case LIBXSMM_RV64_INSTR_GP_MULHSU:
    case LIBXSMM_RV64_INSTR_GP_DIV:
    case LIBXSMM_RV64_INSTR_GP_DIVW:
    case LIBXSMM_RV64_INSTR_GP_DIVU:
    case LIBXSMM_RV64_INSTR_GP_REM:
    case LIBXSMM_RV64_INSTR_GP_REMW:
    case LIBXSMM_RV64_INSTR_GP_REMU:
    case LIBXSMM_RV64_INSTR_GP_REMUW:
    case LIBXSMM_RV64_INSTR_GP_FADD_S:
    case LIBXSMM_RV64_INSTR_GP_FADD_D:
    case LIBXSMM_RV64_INSTR_GP_FSUB_S:
    case LIBXSMM_RV64_INSTR_GP_FSUB_D:
    case LIBXSMM_RV64_INSTR_GP_FMUL_S:
    case LIBXSMM_RV64_INSTR_GP_FMUL_D:
    case LIBXSMM_RV64_INSTR_GP_FDIV_S:
    case LIBXSMM_RV64_INSTR_GP_FDIV_D:
    case LIBXSMM_RV64_INSTR_GP_FSQRT_S:
    case LIBXSMM_RV64_INSTR_GP_FSQRT_D:
    case LIBXSMM_RV64_INSTR_GP_FMADD_S:
    case LIBXSMM_RV64_INSTR_GP_FMADD_D:
    case LIBXSMM_RV64_INSTR_GP_FMSUB_S:
    case LIBXSMM_RV64_INSTR_GP_FMSUB_D:
    case LIBXSMM_RV64_INSTR_GP_FNMADD_S:
    case LIBXSMM_RV64_INSTR_GP_FNMADD_D:
    case LIBXSMM_RV64_INSTR_GP_FNMSUB_S:
    case LIBXSMM_RV64_INSTR_GP_FNMSUB_D:
    case LIBXSMM_RV64_INSTR_GP_FMIN_S:
    case LIBXSMM_RV64_INSTR_GP_FMIN_D:
    case LIBXSMM_RV64_INSTR_GP_FMAX_S:
    case LIBXSMM_RV64_INSTR_GP_FMAX_D:
    case LIBXSMM_RV64_INSTR_GP_VFADD_VF:
      break;
    default:
      fprintf(stderr, "libxsmm_rv64_instruction_alu_compute: unexpected instruction number: %u\n", i_alu_instr);
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
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_1, LIBXSMM_RV64_INSTR_FIELD_RS1);
    /* setting RS1 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_2, LIBXSMM_RV64_INSTR_FIELD_RS2);
    /* setting RS2 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 base ISA I-type instructions */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_alu_compute_imm12( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_alu_instr,
                                                    const unsigned int      i_gp_reg_src,
                                                    const unsigned int      i_gp_reg_dst,
                                                    const int               i_imm12 ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm12: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_2(i_gp_reg_src, i_gp_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_compute_imm12: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_imm12 > 0xfff ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm12: unexpected imm: %u %u\n", i_alu_instr, i_imm12);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* Sanity check */
  switch ( i_alu_instr ) {
    case LIBXSMM_RV64_INSTR_GP_ADDI:
    case LIBXSMM_RV64_INSTR_GP_ADDIW:
    case LIBXSMM_RV64_INSTR_GP_CSRRCI:
    case LIBXSMM_RV64_INSTR_GP_CSRRSI:
    case LIBXSMM_RV64_INSTR_GP_CSRRWI:
    case LIBXSMM_RV64_INSTR_GP_ORI:
    case LIBXSMM_RV64_INSTR_GP_ANDI:
    case LIBXSMM_RV64_INSTR_GP_XORI:
    case LIBXSMM_RV64_INSTR_GP_SRLI:
    case LIBXSMM_RV64_INSTR_GP_SRLIW:
    case LIBXSMM_RV64_INSTR_GP_SLLI:
    case LIBXSMM_RV64_INSTR_GP_SLLIW:
    case LIBXSMM_RV64_INSTR_GP_SLTI:
    case LIBXSMM_RV64_INSTR_GP_SLTIU:
    case LIBXSMM_RV64_INSTR_GP_SRAI:
    case LIBXSMM_RV64_INSTR_GP_SRAIW:
      break;
    default:
      fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm12: unexpected instruction number: %u\n", i_alu_instr);
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
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);
    /* setting RS1 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src, LIBXSMM_RV64_INSTR_FIELD_RS1);
    /* setting IMM12 */
    code[code_head] |= (unsigned int)FILL_REGID(i_imm12, LIBXSMM_RV64_INSTR_FIELD_IMM12);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm12: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 base ISA U-type instructions */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_alu_compute_imm20( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_alu_instr,
                                                    const unsigned int      i_gp_reg_dst,
                                                    const unsigned int      i_imm20 ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm20: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_1(i_gp_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm20: invalid register id %d !\n", i_gp_reg_dst);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_imm20 > 0xfffff ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm20: unexpected imm: %u %u\n", i_alu_instr, i_imm20);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* Sanity check */
  switch ( i_alu_instr ) {
    case LIBXSMM_RV64_INSTR_GP_AUIPC:
    case LIBXSMM_RV64_INSTR_GP_LUI:
      break;
    default:
      fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm20: unexpected instruction number: %u\n", i_alu_instr);
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
  code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);
  /* setting RS1 */
  code[code_head] |= (unsigned int)FILL_REGID(i_imm20, LIBXSMM_RV64_INSTR_FIELD_IMM20);

  /* advance code head */
  io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm20: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

/* RV64 base ISA U-type instructions */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_alu_move_imm12( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_gp_reg_dst,
                                               const int               i_imm12 )
{
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_move_imm12: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* check for imm being in range */
  if ( i_imm12 > 0xfff ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_move_imm12: unexpected imm: %u \n", (unsigned int)i_imm12);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* check that all regs are either 32 or 64 bit */
  if (!REG_VALID_1(i_gp_reg_dst)) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_move_imm12: invalid regsiters id: %d\n", i_gp_reg_dst);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* ADDI immediate to X0 register */
  libxsmm_rv64_instruction_alu_compute_imm12(io_generated_code,
    LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_X0, i_gp_reg_dst, i_imm12);
}

/* RV64 base ISA U-type instructions */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_alu_move_imm20( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_gp_reg_dst,
                                               const unsigned int      i_imm20 ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm20: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_1(i_gp_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_move_imm20: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_imm20 > 0xfffff ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_move_imm20: unexpected imm: %u\n", i_imm20);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* ADDI immediate to X0 register */
  libxsmm_rv64_instruction_alu_compute_imm20(io_generated_code,
    LIBXSMM_RV64_INSTR_GP_LUI, i_gp_reg_dst, i_imm20);

  /* SHIFT Right */
  libxsmm_rv64_instruction_alu_compute_imm12(io_generated_code,
      LIBXSMM_RV64_INSTR_GP_SRLI, i_gp_reg_dst, i_gp_reg_dst, 12);
}

/* Auxilary instruction */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_alu_move_imm32( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_gp_reg_dst,
                                               const unsigned int      i_imm32 ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm32: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_1(i_gp_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_move_imm32: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_imm32 > 0xffffffff ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_move_imm20: unexpected imm: %u\n", i_imm32);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if (i_imm32 <= 0xfff) {
    libxsmm_rv64_instruction_alu_move_imm12( io_generated_code, i_gp_reg_dst, i_imm32 );
  } else if ( i_imm32 <= 0xfffff ){
    libxsmm_rv64_instruction_alu_move_imm20( io_generated_code, i_gp_reg_dst, i_imm32 );
  } else {
#define IMM_12_1 (0xfff)
#define IMM_20_1 (0xfffff000)

   unsigned int imm_12_1 = (i_imm32 & IMM_12_1);
   unsigned int imm_20_1 = ((i_imm32 & IMM_20_1) >> 12);

   /* LUI 20 bits */
   libxsmm_rv64_instruction_alu_compute_imm20(io_generated_code,
       LIBXSMM_RV64_INSTR_GP_LUI, i_gp_reg_dst, imm_20_1);
   /* ADD 12 bits */
   libxsmm_rv64_instruction_alu_compute_imm12(io_generated_code,
       LIBXSMM_RV64_INSTR_GP_ADDI, i_gp_reg_dst, i_gp_reg_dst, imm_12_1);

#undef IMM_12_1
#undef IMM_20_1
  }
}

/* 64 bit immediate move using addi, lui, and shift instructions. */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_alu_set_imm64( libxsmm_generated_code*  io_generated_code,
                                              const unsigned int      i_gp_reg_dst,
                                              const long long         i_imm64 ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_set_imm64: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_1(i_gp_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_set_imm64: invalid register id\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if (i_imm64 <= 0x7ff) {
    libxsmm_rv64_instruction_alu_move_imm12( io_generated_code, i_gp_reg_dst, (int)i_imm64 );
  } else if ( i_imm64 <= 0xfffff ){
    libxsmm_rv64_instruction_alu_move_imm20( io_generated_code, i_gp_reg_dst, (int)i_imm64 );
  } else if ( i_imm64 <= 0xffffffff) {
    libxsmm_rv64_instruction_alu_move_imm32( io_generated_code, i_gp_reg_dst, (int)i_imm64 );
  } else {
#define IMM_12_2 (0xfff00000000)
#define IMM_20_2 (0xfffff00000000000)

    unsigned int imm_12_2 = ((i_imm64 & IMM_12_2) >> 32);
    unsigned int imm_20_2 = ((i_imm64 & IMM_20_2) >> 44);

    /* LUI 20 bits */
    libxsmm_rv64_instruction_alu_compute_imm20(io_generated_code,
        LIBXSMM_RV64_INSTR_GP_LUI, i_gp_reg_dst, imm_20_2);
    /* ADD 12 bits */
    libxsmm_rv64_instruction_alu_compute_imm12(io_generated_code,
        LIBXSMM_RV64_INSTR_GP_ADDI, i_gp_reg_dst, i_gp_reg_dst, imm_12_2);

#undef IMM_12_2
#undef IMM_20_2

    /* SHIFT Left */
    libxsmm_rv64_instruction_alu_compute_imm12(io_generated_code,
        LIBXSMM_RV64_INSTR_GP_SLLI, i_gp_reg_dst, i_gp_reg_dst, 32);

#define IMM_32 (0xffffffff)

    libxsmm_rv64_instruction_alu_move_imm32( io_generated_code, i_gp_reg_dst, i_imm64 & IMM_32 );

#undef IMM_32
  }
}

/* 64-bit compute with immediate uses 64-bit move and alu instructions. */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_alu_compute_imm64( libxsmm_generated_code*  io_generated_code,
                                                  const unsigned int       i_alu_meta_instr,
                                                  const unsigned int       i_gp_reg_src,
                                                  const unsigned int       i_gp_reg_tmp,
                                                  const unsigned int       i_gp_reg_dst,
                                                  const long long          i_imm64 ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm64: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_3(i_gp_reg_src, i_gp_reg_tmp, i_gp_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_alu_compute_imm64: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

#if 0
  if (i_imm64 <= 0xfff) {
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, i_alu_meta_instr, i_gp_reg_tmp, i_gp_reg_dst, i_imm64 );
  } else if (i_imm64 <= 0xfffff){
    libxsmm_rv64_instruction_alu_move_imm20( io_generated_code, i_gp_reg_tmp, i_imm64 );

    /* reg-reg instruction */
    libxsmm_rv64_instruction_alu_compute( io_generated_code, i_alu_meta_instr,
                                         i_gp_reg_src, i_gp_reg_tmp, i_gp_reg_dst);
  } else {
#endif
  {
    /* move imm64 into the temp register */
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, i_gp_reg_tmp, i_imm64 );

    /* reg-reg instruction */
    libxsmm_rv64_instruction_alu_compute( io_generated_code, i_alu_meta_instr,
                                         i_gp_reg_src, i_gp_reg_tmp, i_gp_reg_dst);
  }
}

/* Conditional jump instructions. */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_cond_jump( libxsmm_generated_code* io_generated_code,
                                          const unsigned int      i_jmp_instr,
                                          const unsigned int      i_gp_reg_src_1,
                                          const unsigned int      i_gp_reg_src_2,
                                          const int               i_imm ) {
  printf("Received imm for jmp = %d %d\n", i_imm, (i_imm > 0xfff));
  fflush(stdout);

  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_cond_jump: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_2(i_gp_reg_src_1, i_gp_reg_src_2) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_cond_jump: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_imm > 0xfff ) {
    fprintf(stderr, "libxsmm_rv64_instruction_cond_jump: unexpected imm: %u %d\n", i_jmp_instr, i_imm);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    switch ( i_jmp_instr ) {
      case LIBXSMM_RV64_INSTR_GP_BEQ:
      case LIBXSMM_RV64_INSTR_GP_BGE:
      case LIBXSMM_RV64_INSTR_GP_BGEU:
      case LIBXSMM_RV64_INSTR_GP_BLT:
      case LIBXSMM_RV64_INSTR_GP_BLTU:
      case LIBXSMM_RV64_INSTR_GP_BNE:
        break;
      default:
        fprintf(stderr, "libxsmm_rv64_instruction_cond_jump: unexpected instruction number: %u\n", i_jmp_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
    }

    if ( io_generated_code->code_type > 1 ) {
      unsigned int code_head = io_generated_code->code_size/4;
      unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

      int i_sign = 0;

      unsigned int a_imm = i_imm;
      unsigned int imm_lo;
      unsigned int imm_hi;

      /* Ensure we have enough space */
      if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
        return;
      }

      if (i_imm < 0) {
        i_sign = 1;
        a_imm = (~abs(i_imm) + 1) & 0xfff;
      }

      /* Generate immediate */
      imm_lo = ((a_imm >> 11) & 0x1) | ((a_imm & 0xf) << 1);
      imm_hi = ((a_imm & 0x3f0) >> 4) | (((i_sign)) << 6);

      /* fix bits */
      code[code_head]  = i_jmp_instr;
      /* setting RS1 */
      code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_1, LIBXSMM_RV64_INSTR_FIELD_RS1);
      /* setting RS2 */
      code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_2, LIBXSMM_RV64_INSTR_FIELD_RS2);
      /* setting IMM12HI */
      code[code_head] |= (unsigned int)FILL_REGID(imm_hi, LIBXSMM_RV64_INSTR_FIELD_BIMM12HI);
      /* setting IMM12LO */
      code[code_head] |= (unsigned int)FILL_REGID(imm_lo, LIBXSMM_RV64_INSTR_FIELD_BIMM12LO);

      /* advance code head */
      io_generated_code->code_size += 4;
    } else {
      /* assembly not supported right now */
      fprintf(stderr, "libxsmm_rv64_instruction_cond_jmp: inline/pure assembly print is not supported!\n");
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }
  }
}

/* RV64 Base UJ-type instructions. */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_jump_and_link( libxsmm_generated_code* io_generated_code,
                                     const unsigned int      i_jmp_instr,
                                     const unsigned int      i_gp_reg_dst,
                                     const unsigned int      i_imm ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_jump_and_link: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_1(i_gp_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_jump_and_link: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_imm > 0xfffff ) {
    fprintf(stderr, "libxsmm_rv64_instruction_jal: unexpected imm: %u %u\n", i_jmp_instr, i_imm);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    switch ( i_jmp_instr ) {
      case LIBXSMM_RV64_INSTR_GP_JAL:
        break;
      default:
        fprintf(stderr, "libxsmm_rv64_instruction_jump_and_link: unexpected instruction number: %u\n", i_jmp_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
    }

    if ( io_generated_code->code_type > 1 ) {
      unsigned int code_head = io_generated_code->code_size/4;
      unsigned int* code     = (unsigned int *)io_generated_code->generated_code;
      unsigned int imm_lo    = 0;
      unsigned int imm_hi    = 0;
      unsigned int imm_f     = 0;

      /* Ensure we have enough space */
      if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
        return;
      }

      /* Generate immediate */
      imm_lo = (((i_imm & 0x7fe) << 9)|((i_imm >> 12) & 0xff));
      imm_hi = ((i_imm & 0x800) >> 2) | ((i_imm & 0x100000) >> 1);
      imm_f  = imm_lo | imm_hi;

      /* fix bits */
      code[code_head]  = i_jmp_instr;
      /* setting RS1 */
      code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);
      /* setting IMM20 */
      code[code_head] |= (unsigned int)FILL_REGID(imm_f, LIBXSMM_RV64_INSTR_FIELD_IMM20);

      /* advance code head */
      io_generated_code->code_size += 4;
    } else {
      /* assembly not supported right now */
      fprintf(stderr, "libxsmm_rv64_instruction_jump_and_link: inline/pure assembly print is not supported!\n");
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }
  }
}

/* RV64 Base uncontional jump instructions. */
LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_jump_and_link_reg( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_jmp_instr,
                                              const unsigned int      i_gp_reg_dst,
                                              const unsigned int      i_gp_reg_src_1,
                                              const int               i_imm12 ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_jump_and_link_reg: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_2(i_gp_reg_src_1, i_gp_reg_dst) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_jump_and_link_reg: invalid register!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_imm12 > 0xfff ) {
    fprintf(stderr, "libxsmm_rv64_instruction_jalr: unexpected imm: %u %u\n", i_jmp_instr, i_imm12);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_jmp_instr ) {
    case LIBXSMM_RV64_INSTR_GP_JALR:
      break;
    default:
      fprintf(stderr, "libxsmm_rv64_instruction_jump_and_link_reg: unexpected instruction number: %u\n", i_jmp_instr);
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
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_dst, LIBXSMM_RV64_INSTR_FIELD_RD);
    /* setting RS2 */
    code[code_head] |= (unsigned int)FILL_REGID(i_gp_reg_src_1, LIBXSMM_RV64_INSTR_FIELD_RS1);
    /* setting IMM12HI */
    code[code_head] |= (unsigned int)FILL_REGID(i_imm12, LIBXSMM_RV64_INSTR_FIELD_IMM12);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_jump_and_link_reg: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_register_jump_label( libxsmm_generated_code*     io_generated_code,
                                                    const unsigned int          i_label_no,
                                                    libxsmm_jump_label_tracker* io_jump_label_tracker ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_register_jump_label: at least RV64 needs to be specified as target arch!\n");
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
    fprintf(stderr, "libxsmm_rv64_instruction_register_jump_back_label: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_cond_jump_to_label( libxsmm_generated_code*     io_generated_code,
                                                   const unsigned int          i_jmp_instr,
                                                   const unsigned int          i_gp_reg_src_1,
                                                   const unsigned int          i_gp_reg_src_2,
                                                   const unsigned int          i_label_no,
                                                   libxsmm_jump_label_tracker* io_jump_label_tracker ) {
  unsigned int l_pos;

  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_cond_jump_to_label: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_2(i_gp_reg_src_1, i_gp_reg_src_2) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_cond_jump_to_label: invalid register!\n");
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
    case LIBXSMM_RV64_INSTR_GP_BEQ:
    case LIBXSMM_RV64_INSTR_GP_BGE:
    case LIBXSMM_RV64_INSTR_GP_BGEU:
    case LIBXSMM_RV64_INSTR_GP_BLT:
    case LIBXSMM_RV64_INSTR_GP_BLTU:
    case LIBXSMM_RV64_INSTR_GP_BNE:
      break;
    default:
      fprintf(stderr, "libxsmm_rv64_instruction_cond_jump_back_to_label: unexpected instruction number: %u\n", i_jmp_instr);
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
    unsigned int code_head = io_generated_code->code_size / 4;
    int l_jmp_imm = (l_jmp_dst == 0) /* computing jump immediate */
      ? 0 : (int)l_jmp_dst - (int)code_head;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    if (l_jmp_imm > 0xfff) {
      fprintf(stderr, "libxsmm_rv64_instruction_cond_jump_back_to_label: unexpected jump offser: %u\n", l_jmp_imm);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }

    /* TODO: remove devision here and move to the jump */
    libxsmm_rv64_instruction_cond_jump(io_generated_code, i_jmp_instr,
        /*i_gp_reg_src_1, i_gp_reg_src_2, l_jmp_imm );*/
        i_gp_reg_src_1, i_gp_reg_src_2, l_jmp_imm/2 );

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_cond_jump_to_label: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_register_jump_back_label( libxsmm_generated_code*     io_generated_code,
                                                           libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_register_jump_back_label: at least RV64 needs to be specified as target arch!\n");
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

    printf("Added a new label\n");
    printf("Label count %d\n", io_loop_label_tracker->label_count);
    printf("Label address %d\n", io_loop_label_tracker->label_address[l_lab]);
    fflush(stdout);
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_register_jump_back_label: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_rv64_instruction_cond_jump_back_to_label( libxsmm_generated_code*     io_generated_code,
                                                          const unsigned int          i_jmp_instr,
                                                          const unsigned int          i_gp_reg_src_1,
                                                          const unsigned int          i_gp_reg_src_2,
                                                          libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  if ( io_generated_code->arch < LIBXSMM_RV64 ) {
    fprintf(stderr, "libxsmm_rv64_instruction_cond_jump_back_to_label: at least RV64 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( !REG_VALID_2(i_gp_reg_src_1, i_gp_reg_src_2) ) {
    fprintf(stderr, "libxsmm_rv64_instruction_cond_jump_back_to_lable: invalid register id !\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_jmp_instr ) {
    case LIBXSMM_RV64_INSTR_GP_BEQ:
    case LIBXSMM_RV64_INSTR_GP_BGE:
    case LIBXSMM_RV64_INSTR_GP_BGEU:
    case LIBXSMM_RV64_INSTR_GP_BLT:
    case LIBXSMM_RV64_INSTR_GP_BLTU:
    case LIBXSMM_RV64_INSTR_GP_BNE:
      break;
    default:
      fprintf(stderr, "libxsmm_rv64_instruction_cond_jump_back_to_label: unexpected instruction number: %u\n", i_jmp_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_lab = --io_loop_label_tracker->label_count;
    unsigned int l_jmp_dst = (io_loop_label_tracker->label_address[l_lab]);
    unsigned int code_head = io_generated_code->code_size;
    int l_jmp_imm = (int)l_jmp_dst - (int)code_head; /* computing jump immediate */

    printf("Jump address %x %x\n", l_jmp_dst, code_head);
    fflush(stdout);

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    if (l_jmp_imm > 0xfff) {
      fprintf(stderr, "libxsmm_rv64_instruction_cond_jump_back_to_label: unexpected jump offser: %u\n", l_jmp_imm);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }

    libxsmm_rv64_instruction_cond_jump(io_generated_code, i_jmp_instr,
        i_gp_reg_src_1, i_gp_reg_src_2, l_jmp_imm/2);
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_rv64_instruction_cond_jump_back_to_label: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

#undef FILL_REGID
#undef REG_VALID_1
#undef REG_VALID_2
#undef REG_VALID_3
