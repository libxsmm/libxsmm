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
#include "generator_aarch64_instructions.h"


LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_open_stream( libxsmm_generated_code* io_generated_code,
                                              const unsigned short    i_callee_save_bitmask ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_close_stream: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
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
void libxsmm_aarch64_instruction_restore_regs( libxsmm_generated_code* io_generated_code,
                                               const unsigned short    i_callee_save_bitmask ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_restore_regs: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
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
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_close_stream( libxsmm_generated_code* io_generated_code,
                                               const unsigned short    i_callee_save_bitmask ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_close_stream: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  libxsmm_aarch64_instruction_restore_regs( io_generated_code, i_callee_save_bitmask );

  /* generate return instruction */
  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* insert ret instruction */
    code[code_head] = 0xd65f03c0;

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_close_stream: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_close_data( libxsmm_generated_code*     io_generated_code,
                                             libxsmm_const_data_tracker* io_const_data ) {
  unsigned int l_i;
  unsigned char* l_code_buffer = (unsigned char*) io_generated_code->generated_code;
  unsigned int l_code_size = io_generated_code->code_size;
  unsigned int l_data_size = io_const_data->const_data_size;
  unsigned int l_max_size = io_generated_code->buffer_size;

  /* Handle any constant data */
  if ( l_data_size > 0 ) {
    /* Round up to a page boundary */
    l_code_size = LIBXSMM_UP( l_code_size, LIBXSMM_PAGE_MINSIZE ); /* Check me */

    /* Ensure we have space in the code stream */
    if ( l_max_size < l_data_size + l_code_size ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* Copy the data into the buffer */
    memcpy( l_code_buffer + l_code_size, io_const_data->const_data, l_data_size );

    /* Update the data size including unused space (page-size alignment */
    io_generated_code->data_size = l_code_size + l_data_size - io_generated_code->code_size;

    /* Fill in the load address */
    for ( l_i = 0; l_i < io_const_data->const_data_nload_insns; l_i++ ) {
      unsigned int l_adr_off = io_const_data->const_data_pc_load_insns[l_i];
      unsigned int l_off, l_gp, l_pc_off, l_insn;

      /* Read the user-provided offset and destination GP */
      memcpy( &l_off, l_code_buffer + l_adr_off, sizeof(l_off) );

      /* Extract the GP from the top 5 bits */
      l_gp = l_off >> 27;

      /* Compute the PC offset */
      l_pc_off = l_code_size - l_adr_off + (0x1fffff & l_off);

      /* Construct the final ADR instruction */
      l_insn  = (0x1 << 28) | (0x1f & l_gp);
      l_insn |= ((0x1ffffc & l_pc_off) << 3) | ((0x3 & l_pc_off) << 29);
      memcpy( l_code_buffer + l_adr_off, &l_insn, sizeof(l_insn) );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_adr_data( libxsmm_generated_code*     io_generated_code,
                                           unsigned int                i_reg,
                                           unsigned int                i_off,
                                           libxsmm_const_data_tracker* io_const_data ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size / 4;
    unsigned int* code     = (unsigned int*) io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size + 4 < io_generated_code->code_size ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* Ensure we have space in the fixup buffer */
    if ( 128 <= io_const_data->const_data_nload_insns ) {
      fprintf( stderr, "libxsmm_aarch64_instruction_adr_data out of fixup space!\n" );
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }

    /* Save the offset and register in the code */
    code[code_head] = ((0x1f & i_reg) << 27) | (0x1fffff & i_off);

    /* Save the adr offset */
    io_const_data->const_data_pc_load_insns[io_const_data->const_data_nload_insns++] = io_generated_code->code_size;

    /* Advance code head */
    io_generated_code->code_size += 4;
  } else {
    fprintf(stderr, "libxsmm_aarch64_instruction_adr_data: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_aarch64_instruction_add_data( libxsmm_generated_code*     io_generated_code,
                                                   const unsigned char*        i_data,
                                                   unsigned int                i_ndata_bytes,
                                                   unsigned int                i_alignment,
                                                   unsigned int                i_append_only,
                                                   libxsmm_const_data_tracker* io_const_data ) {
  i_alignment = LIBXSMM_MAX( i_alignment, 1 );

  if ( io_generated_code->code_type > 1 ) {
    unsigned char* l_data = (unsigned char*) io_const_data->const_data;
    unsigned int l_dsize = io_const_data->const_data_size;
    unsigned int l_doff, l_npad;

    /* See if we already have the data */
    if ( !i_append_only ) {
      for ( l_doff = 0; l_doff < l_dsize; l_doff += i_alignment ) {
        if ( i_ndata_bytes <= l_dsize - l_doff && !memcmp( l_data + l_doff, i_data, i_ndata_bytes) ) {
          return l_doff;
        }
      }
    }

    /* Determine how much padding is needed */
    l_npad = LIBXSMM_UP( l_dsize, i_alignment) - l_dsize;

    /* Ensure we have enough space */
    if ( ((size_t)l_dsize + l_npad + i_ndata_bytes) > sizeof(io_const_data->const_data) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return ~0U;
    }

    /* Copy the data */
    memcpy( l_data + l_dsize + l_npad, i_data, i_ndata_bytes );

    /* Update the size */
    io_const_data->const_data_size += l_npad + i_ndata_bytes;

    /* Return the offset of the new data in the buffer */
    return l_dsize + l_npad;
  } else {
    fprintf(stderr, "libxsmm_aarch64_instruction_add_data: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return 0;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_move( libxsmm_generated_code*           io_generated_code,
                                             const unsigned int                i_vmove_instr,
                                             const unsigned int                i_gp_reg_addr,
                                             const unsigned int                i_gp_reg_offset,
                                             const int                         i_offset,
                                             const unsigned int                i_vec_reg,
                                             const libxsmm_aarch64_asimd_width i_asimdwidth ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_move: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
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
        int l_imm;
        /* adjust offset based on vector width */
        switch ( (int)i_asimdwidth ) {
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
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }
        code[code_head] |= (unsigned int)((0x00000fff & l_imm) << 10);
      } else {
        if ( (i_offset < -256) || (i_offset > 255) ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_asimd_move: offset for per-index/post-index addressing mode out of range: %i!\n", i_offset);
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }
        code[code_head] |= (unsigned int)((0x000001ff & i_offset) << 12);
      }
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_move: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_gpr_move( libxsmm_generated_code*           io_generated_code,
                                                 const unsigned int                i_vmove_instr,
                                                 const unsigned int                i_gp_reg,
                                                 const unsigned int                i_vec_reg,
                                                 const short                       i_index,
                                                 const libxsmm_aarch64_asimd_width i_asimdwidth ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_gpr_move: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_vmove_instr ) {
    case LIBXSMM_AARCH64_INSTR_ASIMD_UMOV_V_G:
    case LIBXSMM_AARCH64_INSTR_ASIMD_MOV_G_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_DUP_HALF:
    case LIBXSMM_AARCH64_INSTR_ASIMD_DUP_FULL:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_asimd_gpr_move: unexpected instruction number: %u\n", i_vmove_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int*)io_generated_code->generated_code;
    unsigned int l_imm5 = 0x0;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* fix bits */
    code[code_head] = (unsigned int)(0xffffff00 & i_vmove_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)( (i_vmove_instr & 0x8) == 0x8 ? (0x1f & i_vec_reg) : (0x1f & i_gp_reg) );
    /* setting Rn */
    code[code_head] |= (unsigned int)( (i_vmove_instr & 0x8) == 0x8 ? ((0x1f & i_gp_reg) << 5) : ((0x1f & i_vec_reg) << 5) );
    /* setting Q */
    code[code_head] |= (unsigned int)( ( i_asimdwidth == LIBXSMM_AARCH64_ASIMD_WIDTH_D ) ? 0x40000000 : 0x0 );

    if (i_vmove_instr == LIBXSMM_AARCH64_INSTR_ASIMD_DUP_FULL) {
      code[code_head] |= (unsigned int) 0x40000000;
    }

    /* setting imm5 */
    if ( i_asimdwidth == LIBXSMM_AARCH64_ASIMD_WIDTH_B ) {
      l_imm5 = 0x1 | ((i_index & 0xf) << 1);
    } else if ( i_asimdwidth == LIBXSMM_AARCH64_ASIMD_WIDTH_H ) {
      l_imm5 = 0x2 | ((i_index & 0x7) << 2);
    } else if ( i_asimdwidth == LIBXSMM_AARCH64_ASIMD_WIDTH_S ) {
      l_imm5 = 0x4 | ((i_index & 0x3) << 3);
    } else if ( i_asimdwidth == LIBXSMM_AARCH64_ASIMD_WIDTH_D ) {
      l_imm5 = 0x8 | ((i_index & 0x1) << 4);
    } else {
      /* should not happen */
      fprintf(stderr, "libxsmm_aarch64_instruction_asimd_gpr_move: unexpected datatype for instruction: %u\n", i_vmove_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }
    /* setting imm5 */
    code[code_head] |= (unsigned int)( l_imm5 << 16 );

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_gpr_move: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_struct_r_move( libxsmm_generated_code*               io_generated_code,
                                                      const unsigned int                    i_vmove_instr,
                                                      const unsigned int                    i_gp_reg_addr,
                                                      const unsigned int                    i_gp_reg_offset,
                                                      const unsigned int                    i_vec_reg,
                                                      const libxsmm_aarch64_asimd_tupletype i_tupletype ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_r_move: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_vmove_instr ) {
    case LIBXSMM_AARCH64_INSTR_ASIMD_LD1R:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LD1_1:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LD1_2:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LD1_3:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LD1_4:
        case LIBXSMM_AARCH64_INSTR_ASIMD_ST1_1:
    case LIBXSMM_AARCH64_INSTR_ASIMD_ST1_2:
    case LIBXSMM_AARCH64_INSTR_ASIMD_ST1_3:
    case LIBXSMM_AARCH64_INSTR_ASIMD_ST1_4:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_r_move: unexpected instruction number: %u\n", i_vmove_instr);
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
    code[code_head] = (unsigned int)(0xffffff00 & i_vmove_instr);
    /* setting Rt */
    code[code_head] |= (unsigned int)(0x1f & i_vec_reg);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_addr) << 5);
    /* setting size */
    code[code_head] |= (unsigned int)((0x6 & (unsigned int)i_tupletype) << 9);
    /* setting opc */
    code[code_head] |= (unsigned int)((0x1 & (unsigned int)i_tupletype) << 30);

    /* load/store with offset register */
    if ( (i_vmove_instr & 0x3) == 0x3 && ((i_vmove_instr & 0xff000000) == 0x0d000000)) {
      /* setting Rm */
      code[code_head] |= (unsigned int)((0x1f & i_gp_reg_offset) << 16);
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_r_move: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_struct_move( libxsmm_generated_code*           io_generated_code,
                                                    const unsigned int                i_vmove_instr,
                                                    const unsigned int                i_gp_reg_addr,
                                                    const unsigned int                i_gp_reg_offset,
                                                    const int                         i_offset,
                                                    const unsigned int                i_vec_reg,
                                                    const short                       i_index,
                                                    const libxsmm_aarch64_asimd_width i_asimdwidth ) {
  LIBXSMM_UNUSED( i_offset );

  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_move: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_vmove_instr ) {
    case LIBXSMM_AARCH64_INSTR_ASIMD_LD1_I_POST:
    case LIBXSMM_AARCH64_INSTR_ASIMD_LD1_R_POST:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_move: unexpected instruction number: %u\n", i_vmove_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int*)io_generated_code->generated_code;
    unsigned int l_q = 0, l_s = 0, l_sz = 0;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* fix bits */
    code[code_head] = (unsigned int)(0xffffff00 & i_vmove_instr);
    /* setting Rt */
    code[code_head] |= (unsigned int)(0x1f & i_vec_reg);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_addr) << 5);
    /* setting Rm */
    if ( (0x3 & i_vmove_instr) == 0x3 ) {
      code[code_head] |= (unsigned int)((0x1f & i_gp_reg_offset) << 16);
    }

    switch ( (int)i_asimdwidth ) {
      case LIBXSMM_AARCH64_ASIMD_WIDTH_S:
        l_q = 0x1 & (i_index >> 1);
        l_s = 0x1 & (i_index >> 0);

        if ( (0x3 & i_vmove_instr) != 0x3 && i_offset != 4 ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_move: unexpected i_offset: %d\n", i_offset);
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }
        break;
      case LIBXSMM_AARCH64_ASIMD_WIDTH_D:
        l_q = 0x1 & i_index;
        l_sz = 0x1;

        if ( (0x3 & i_vmove_instr) != 0x3 && i_offset != 8 ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_move: unexpected i_offset: %d\n", i_offset);
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }
        break;
      default:
        fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_move: unexpected asimdwidth number: %u\n", i_asimdwidth);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
    }

    /* setting Q */
    code[code_head] |= (unsigned int)(l_q << 30);
    /* setting S */
    code[code_head] |= (unsigned int)(l_s << 12);
    /* setting size */
    code[code_head] |= (unsigned int)(l_sz << 10);

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_struct_move: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_pair_move( libxsmm_generated_code*           io_generated_code,
                                                  const unsigned int                i_vmove_instr,
                                                  const unsigned int                i_gp_reg_addr,
                                                  const int                         i_offset,
                                                  const unsigned int                i_vec_reg_0,
                                                  const unsigned int                i_vec_reg_1,
                                                  const libxsmm_aarch64_asimd_width i_asimdwidth ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_pair_move: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
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
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;
    unsigned char l_opc = 0x0;
    signed char l_imm = 0x0;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    switch ( (int)i_asimdwidth ) {
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
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
    }

    if ( (l_imm < -64) || (l_imm > 63) ) {
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: offset out of range: %i!\n", i_offset);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_asimd_compute( libxsmm_generated_code*               io_generated_code,
                                                const unsigned int                    i_vec_instr,
                                                const unsigned int                    i_vec_reg_src_0,
                                                const unsigned int                    i_vec_reg_src_1,
                                                const unsigned char                   i_idx_shf,
                                                const unsigned int                    i_vec_reg_dst,
                                                const libxsmm_aarch64_asimd_tupletype i_tupletype ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_compute: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_vec_instr ) {
    case LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_ORR_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_AND_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_ADD_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_ADDV_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_BIC_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_BIF_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_BIT_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_BSL_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_NEG_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_NOT_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_ORN_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_SHL_I_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_SSHR_I_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_USHR_I_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_SSHL_R_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_USHL_R_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_CMEQ_R_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_CMEQ_Z_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_CMGE_R_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_CMGE_Z_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_CMGT_R_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_CMGT_Z_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_CMLE_Z_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_CMLT_Z_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_S:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMLS_E_S:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMLS_E_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMLS_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FADD_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FSUB_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMUL_E_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FDIV_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FNEG_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FSQRT_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FRECPE_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FRECPS_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FRSQRTE_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FRSQRTS_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMAX_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMIN_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FADDP_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMAXP_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FMINP_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FCMEQ_R_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FCMEQ_Z_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FCMGE_R_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FCMGE_Z_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_R_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FCMGT_Z_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FCMLE_Z_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FCMLT_Z_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FRINTM_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_FCVTMS_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_TRN1:
    case LIBXSMM_AARCH64_INSTR_ASIMD_TRN2:
    case LIBXSMM_AARCH64_INSTR_ASIMD_ZIP1:
    case LIBXSMM_AARCH64_INSTR_ASIMD_ZIP2:
    case LIBXSMM_AARCH64_INSTR_ASIMD_UZP1:
    case LIBXSMM_AARCH64_INSTR_ASIMD_UZP2:
    case LIBXSMM_AARCH64_INSTR_ASIMD_TBL_1:
    case LIBXSMM_AARCH64_INSTR_ASIMD_TBL_2:
    case LIBXSMM_AARCH64_INSTR_ASIMD_TBL_3:
    case LIBXSMM_AARCH64_INSTR_ASIMD_TBL_4:
    case LIBXSMM_AARCH64_INSTR_ASIMD_TBX_1:
    case LIBXSMM_AARCH64_INSTR_ASIMD_TBX_2:
    case LIBXSMM_AARCH64_INSTR_ASIMD_TBX_3:
    case LIBXSMM_AARCH64_INSTR_ASIMD_TBX_4:
    case LIBXSMM_AARCH64_INSTR_ASIMD_BFMMLA_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_SMMLA_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_UMMLA_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_USMMLA_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_BFDOT_E_V:
    case LIBXSMM_AARCH64_INSTR_ASIMD_BFDOT_V:
       break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_asimd_compute: unexpected instruction number: 0x%08x\n", i_vec_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( ((0x3 & i_vec_instr) == 2) && (i_vec_reg_src_1 != LIBXSMM_AARCH64_ASIMD_REG_UNDEF) ) {
      fprintf(stderr, "libxsmm_aarch64_instruction_asimd_compute: got 3 registers, but instruction has only 2: 0x%08x\n", i_vec_instr);
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
    code[code_head]  = (unsigned int)(0xffffff00 & i_vec_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)(0x1f & i_vec_reg_dst);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_vec_reg_src_0) << 5);
    /* setting Rm */
    if ( (0x3 & i_vec_instr) == 0x3 ) {
      code[code_head] |= (unsigned int)((0x1f & i_vec_reg_src_1) << 16);
    }
    /* setting Q */
    code[code_head] |= (unsigned int)((0x1 & i_tupletype) << 30);
    /* setting sz */
    if ( (0x10 & i_vec_instr) == 0x0 ) {
      if ( (0x8 & i_vec_instr) == 0x8 ) {
        code[code_head] |= (unsigned int)((0x2 & i_tupletype) << 21);
      } else {
        code[code_head] |= (unsigned int)((0x6 & i_tupletype) << 21);
      }
    }
    /* FMLA, eltwise */
    if ( ((0x4 & i_vec_instr) == 0x4) && ((0x18 & i_vec_instr) != 0x18) ) {
      unsigned char l_idx = (unsigned char)(( i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D ) ? i_idx_shf << 1 : i_idx_shf);
      if ( (i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D && i_idx_shf > 2) || (i_idx_shf > 4) ) {
        fprintf(stderr, "libxsmm_aarch64_instruction_asimd_compute: incompatible tuple and index type for fmla instruction: 0x%08x\n", i_vec_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }

      /* setting L */
      code[code_head] |= (unsigned int)((0x1 & l_idx) << 21);
      /* setting H */
      code[code_head] |= (unsigned int)((0x2 & l_idx) << 10);
    }
    /* shifts */
    if ( (0x18 & i_vec_instr) == 0x18 ) {
      unsigned char l_shift = 0x0;
      if ( ( i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B ) || ( i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B ) ) {
        l_shift = (unsigned char)(0x8 | (((0x4 & i_vec_instr) == 0x4) ? (0x8 - (i_idx_shf & 0x7)) : (i_idx_shf & 0x7) ));
      } else if ( ( i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4H ) || ( i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8H ) ) {
        l_shift = (unsigned char)(0x10 | (((0x4 & i_vec_instr) == 0x4) ? (0x10 - (i_idx_shf & 0xf)) : (i_idx_shf & 0xf) ));
      } else if ( ( i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2S ) || ( i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S ) ) {
        l_shift = (unsigned char)(0x20 | (((0x4 & i_vec_instr) == 0x4) ? (0x20 - (i_idx_shf & 0x1f)) : (i_idx_shf & 0x1f) ));
      } else if ( i_tupletype == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D ) {
        l_shift = (unsigned char)(0x40 | (((0x4 & i_vec_instr) == 0x4) ? (0x40 - (i_idx_shf & 0x3f)) : (i_idx_shf & 0x3f) ));
      } else {
        fprintf(stderr, "libxsmm_aarch64_instruction_asimd_compute: incompatible tuple and index type for shift nstruction: 0x%08x\n", i_vec_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
      code[code_head] |= (unsigned int)(l_shift << 16);
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_asimd_compute: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_move( libxsmm_generated_code*                io_generated_code,
                                           const unsigned int                     i_vmove_instr,
                                           const unsigned int                     i_gp_reg_addr,
                                           const unsigned int                     i_reg_offset_idx,
                                           const int                              i_offset,
                                           const unsigned int                     i_vec_reg,
                                           const unsigned int                     i_pred_reg ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE128 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: at least ARM SVE128 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_vmove_instr ) {
    case LIBXSMM_AARCH64_INSTR_SVE_LDR_P_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_STR_P_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1B_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1B_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1H_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1W_SR:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1D_V_OFF64:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1W_V_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1W_V_OFF_SCALE:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1W_V_OFF64_SCALE:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1H_V_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1H_V_OFF_SCALE:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1H_V_OFF64_SCALE:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1H_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1D_SR:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1D_SR:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1D_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_STNT1D_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1W_SR:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1W_V_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1W_V_OFF_SCALE:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1W_V_OFF64_SCALE:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1H_V_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1H_V_OFF_SCALE:
    case LIBXSMM_AARCH64_INSTR_SVE_ST1H_V_OFF64_SCALE:
    case LIBXSMM_AARCH64_INSTR_SVE_STNT1W_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1RB_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1RH_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_LD1RQD_I_OFF:
       break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: unexpected instruction number: %x\n", i_vmove_instr);
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
      code[code_head] |= (unsigned int)((0x1f & i_reg_offset_idx) << 16);
    }
    if ( (i_vmove_instr & 0x4) == 0x4 ) {
      if ( (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF) || (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF) ) {
        unsigned char l_offset = 0;
        if ( i_offset < 0 ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: for LD1RW, LD1RD only positive offsets are allowed!\n");
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }

        if ( i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF ) {
          l_offset = (unsigned char)((unsigned int)i_offset >> 2);
        } else if ( i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF ) {
          l_offset = (unsigned char)((unsigned int)i_offset >> 3);
        }

        if ( l_offset > 63 ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: for LD1RW, LD1RD offset is out of range!\n");
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }

        code[code_head] |= (unsigned int)((0x3f & l_offset) << 16);
      } else if ( (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LDR_P_I_OFF) || (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF) ||
                  (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_STR_P_I_OFF) || (i_vmove_instr == LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF)    ) {
        if ( i_offset < -256 || i_offset > 256 ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: for STR/LDR offset is out of range!\n");
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }

        code[code_head] |= (unsigned int)((0x7 & i_offset) << 10);
        code[code_head] |= (unsigned int)((0x3f & (i_offset >> 3)) << 16);
      } else {
        int l_offset = i_offset;

        /* TODO: Make this generic */
        if ( LIBXSMM_AARCH64_INSTR_SVE_LD1RQD_I_OFF == i_vmove_instr ) {
          l_offset /= 16;
        }

        if ( l_offset < -8 || l_offset > 7 ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_sve_move: for LD1W/D, LD1RQD, ST[NT]1W/D, offset is out of range!\n");
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }

        code[code_head] |= (unsigned int)((0xf & l_offset) << 16);
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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_prefetch( libxsmm_generated_code*            io_generated_code,
                                               const unsigned int                 i_prefetch_instr,
                                               const unsigned int                 i_gp_reg_addr,
                                               const unsigned int                 i_gp_reg_offset,
                                               const int                          i_offset,
                                               const unsigned int                 i_pred_reg,
                                               const libxsmm_aarch64_sve_prefetch i_prefetch ) {
  LIBXSMM_UNUSED( i_gp_reg_offset );

  if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE128 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_prefetch: at least ARM SVE128 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_prefetch_instr ) {
    case LIBXSMM_AARCH64_INSTR_SVE_PRFW_I_OFF:
    case LIBXSMM_AARCH64_INSTR_SVE_PRFD_I_OFF:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_sve_prefetch: unexpected instruction number: %u\n", i_prefetch_instr);
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
    code[code_head]  = (unsigned int)(0xffffff00 & i_prefetch_instr);
    /* fix prfop */
    code[code_head] |= (unsigned int)(0xf & i_prefetch);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_addr) << 5);
    /* setting p reg */
    code[code_head] |= (unsigned int)((0x7 & i_pred_reg) << 10);

    /* setting imm6 */
    if ( (i_prefetch_instr & 0x4) == 0x4 ) {
      if ( (i_offset < -32) || (i_offset > 31) ) {
        fprintf(stderr, "libxsmm_aarch64_instruction_sve_prefetch: offset out of range: %d!\n", i_offset);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }

      code[code_head] |= (unsigned int)(i_offset << 16);
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_prefetch: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_compute( libxsmm_generated_code*        io_generated_code,
                                              const unsigned int             i_vec_instr,
                                              const unsigned int             i_vec_reg_src_0,
                                              const unsigned int             i_vec_reg_src_1,
                                              const unsigned char            i_index,
                                              const unsigned int             i_vec_reg_dst,
                                              const unsigned int             i_pred_reg,
                                              const libxsmm_aarch64_sve_type i_type ) {
  unsigned char l_vec_reg_src_0 = LIBXSMM_CAST_UCHAR(i_vec_reg_src_0);
  unsigned char l_vec_reg_src_1 = LIBXSMM_CAST_UCHAR(i_vec_reg_src_1);

  unsigned char l_has_two_sources = (i_vec_instr & LIBXSMM_AARCH64_INSTR_SVE_HAS_SRC1) == LIBXSMM_AARCH64_INSTR_SVE_HAS_SRC1;
  unsigned char l_is_predicated = (i_vec_instr & LIBXSMM_AARCH64_INSTR_SVE_IS_PREDICATED) == LIBXSMM_AARCH64_INSTR_SVE_IS_PREDICATED;
  unsigned char l_is_type_specific =    i_vec_instr != LIBXSMM_AARCH64_INSTR_SVE_EOR_V
                                     && i_vec_instr != LIBXSMM_AARCH64_INSTR_SVE_ORR_V
                                     && i_vec_instr != LIBXSMM_AARCH64_INSTR_SVE_AND_V
                                     && i_vec_instr != LIBXSMM_AARCH64_INSTR_SVE_BFMMLA_V
                                     && i_vec_instr != LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V
                                     && i_vec_instr != LIBXSMM_AARCH64_INSTR_SVE_LSR_I_V;
  unsigned char l_is_indexed = (i_vec_instr & LIBXSMM_AARCH64_INSTR_SVE_IS_INDEXED) == LIBXSMM_AARCH64_INSTR_SVE_IS_INDEXED;
  unsigned char l_has_logical_shift_imm = i_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V || i_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_LSR_I_V; /* a special case for now */
  unsigned char l_has_immediate = (i_vec_instr & LIBXSMM_AARCH64_INSTR_SVE_HAS_IMM) == LIBXSMM_AARCH64_INSTR_SVE_HAS_IMM;
  unsigned int l_vec_instr = i_vec_instr;

  if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE128 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_compute: at least ARM SVE needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* this is a check whether the instruction is valid; it could removed for better performance */
  switch ( i_vec_instr ) {
    case LIBXSMM_AARCH64_INSTR_SVE_MOV_R_P:
    case LIBXSMM_AARCH64_INSTR_SVE_DUP_GP_V:
    case LIBXSMM_AARCH64_INSTR_SVE_ORR_P:
    case LIBXSMM_AARCH64_INSTR_SVE_SEL_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_AND_V:
    case LIBXSMM_AARCH64_INSTR_SVE_EOR_V:
    case LIBXSMM_AARCH64_INSTR_SVE_ORR_V:
    case LIBXSMM_AARCH64_INSTR_SVE_ADD_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FADD_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FSUB_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FMUL_V:
    case LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V:
    case LIBXSMM_AARCH64_INSTR_SVE_LSR_I_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_I:
    case LIBXSMM_AARCH64_INSTR_SVE_FMLS_V_I:
    case LIBXSMM_AARCH64_INSTR_SVE_FMUL_V_I:
    case LIBXSMM_AARCH64_INSTR_SVE_FADD_I_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FMUL_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FDIV_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FDIVR_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_SMAX_V_I:
    case LIBXSMM_AARCH64_INSTR_SVE_SMIN_V_I:
    case LIBXSMM_AARCH64_INSTR_SVE_FMLA_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FMLS_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FNEG_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_BFCVT_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FADDV_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FMAXV_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FMINV_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FRECPS_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FRECPE_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FSQRT_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FRSQRTE_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FRSQRTS_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FRINTM_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FRINTI_V_P:
    case LIBXSMM_AARCH64_INSTR_SVE_FCVTZS_V_P_SS:
    case LIBXSMM_AARCH64_INSTR_SVE_SCVTF_V_P_SS:
    case LIBXSMM_AARCH64_INSTR_SVE_FCMEQ_P_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FCMNE_P_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FCMGT_P_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FCMGE_P_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FCMLT_P_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FCMLE_P_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FCMGT_Z_V:
    case LIBXSMM_AARCH64_INSTR_SVE_CMPGT_Z_V:
    case LIBXSMM_AARCH64_INSTR_SVE_UZP_P_E:
    case LIBXSMM_AARCH64_INSTR_SVE_UZP_P_O:
    case LIBXSMM_AARCH64_INSTR_SVE_ZIP_P_H:
    case LIBXSMM_AARCH64_INSTR_SVE_ZIP_P_L:
    case LIBXSMM_AARCH64_INSTR_SVE_ZIP1_V:
    case LIBXSMM_AARCH64_INSTR_SVE_ZIP2_V:
    case LIBXSMM_AARCH64_INSTR_SVE_UZP1_V:
    case LIBXSMM_AARCH64_INSTR_SVE_UZP2_V:
    case LIBXSMM_AARCH64_INSTR_SVE_TRN1_V:
    case LIBXSMM_AARCH64_INSTR_SVE_TRN2_V:
    case LIBXSMM_AARCH64_INSTR_SVE_UUNPKLO_V:
    case LIBXSMM_AARCH64_INSTR_SVE_UUNPKHI_V:
    case LIBXSMM_AARCH64_INSTR_SVE_TBL:
    case LIBXSMM_AARCH64_INSTR_SVE_TBX:
    case LIBXSMM_AARCH64_INSTR_SVE_SUB_V_I:
    case LIBXSMM_AARCH64_INSTR_SVE_BFMMLA_V:
    case LIBXSMM_AARCH64_INSTR_SVE_FMMLA_V:
    case LIBXSMM_AARCH64_INSTR_SVE_SMMLA_V:
    case LIBXSMM_AARCH64_INSTR_SVE_UMMLA_V:
    case LIBXSMM_AARCH64_INSTR_SVE_USMMLA_V:
    case LIBXSMM_AARCH64_INSTR_SVE_BFDOT_V_I:
    case LIBXSMM_AARCH64_INSTR_SVE_BFDOT_V:
    case LIBXSMM_AARCH64_INSTR_SVE_USDOT_V:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_sve_compute: unexpected instruction number: 0x%08x\n", i_vec_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  /* fp compare less than is a pseudo instruction: greater than or equal with switched source registers */
  if ( l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FCMLT_P_V ) {
    unsigned char l_tmp = l_vec_reg_src_0;
    l_vec_instr = LIBXSMM_AARCH64_INSTR_SVE_FCMGE_P_V;
    l_vec_reg_src_0 = l_vec_reg_src_1;
    l_vec_reg_src_1 = l_tmp;
  }

  /* fp compare less equal than is a pseudo instruction: greater than with switched source registers */
  if ( l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FCMLE_P_V ) {
    unsigned char l_tmp = l_vec_reg_src_0;
    l_vec_instr = LIBXSMM_AARCH64_INSTR_SVE_FCMGT_P_V;
    l_vec_reg_src_0 = l_vec_reg_src_1;
    l_vec_reg_src_1 = l_tmp;
  }

  /* add with immediate is currently the only instruction with an immediate; may be a flag in the future */
  /* this check could be disabled for performance reasons */
  if ( l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FADD_I_P ) {
    if ( l_vec_reg_src_0 > 1) {
      fprintf(stderr, "libxsmm_aarch64_instruction_sve_compute: immediate for FADD may be 0 for 0.5 for 1 for 1.0, but nothing else! Received %x\n", l_vec_reg_src_1 );
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }
  }

  /* special instruction, where only dst = src_0 is supported */
  /* this check could be disabled for performance reasons */
  if ( (l_vec_instr & LIBXSMM_AARCH64_INSTR_SVE_SRC0_IS_DST) == LIBXSMM_AARCH64_INSTR_SVE_SRC0_IS_DST ) {
    if ( i_vec_reg_src_0 != i_vec_reg_dst ) {
      if (i_vec_reg_src_1 == i_vec_reg_dst &&
        (l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P ||
         l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P ||
         l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FMUL_V_P)) {
        /* arguments can be switched around */
        /* we assign 0 <- 1 anyways, so we just skip that */
      } else if (i_vec_reg_src_1 == i_vec_reg_dst &&
        (l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FDIV_V_P ||
         l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FDIVR_V_P)) {
        /* arguments can be switched around + instruction can be exchanged */
        /* we assign 0 <- 1 anyways, so we just skip that */
        l_vec_instr = l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FDIV_V_P ? LIBXSMM_AARCH64_INSTR_SVE_FDIVR_V_P : LIBXSMM_AARCH64_INSTR_SVE_FDIV_V_P;
      } else {
        fprintf(stderr, "libxsmm_aarch64_instruction_sve_compute: instruction 0x%08x only supports i_vec_reg_src_0 == i_vec_reg_dst, but %u != %u\n", i_vec_instr, i_vec_reg_src_0, i_vec_reg_dst);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
    } else if ( l_has_two_sources ||
              (l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FMAX_V_P ||
               l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FMIN_V_P ||
               l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FMUL_V_P ||
               l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FDIV_V_P ||
               l_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_FDIVR_V_P ) ) {
      l_vec_reg_src_0 = l_vec_reg_src_1;
    }
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *) io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* fix bits, 0x10 must not be used as flags */
    code[code_head] = (unsigned int)(0xffffff10 & l_vec_instr);
    /* setting Zda/Zdn */
    code[code_head] |= (unsigned int)(0x1f & i_vec_reg_dst);
    /* setting Zn */
    code[code_head] |= (unsigned int)((0x1f & l_vec_reg_src_0) << 5);
    if ( l_has_logical_shift_imm ) {
      unsigned char l_elementSizeBits = (unsigned char)(8 << (int)i_type); /* B -> 8, H -> 16, S -> 32, D -> 64 */
     /* the encoding for right shift is reversed */
      unsigned char l_index = i_vec_instr == LIBXSMM_AARCH64_INSTR_SVE_LSL_I_V ? i_index : LIBXSMM_MAX( l_elementSizeBits - i_index, 0 );
      /* left/right shift (immediate) has a special encoding, which was not used before */
      unsigned int l_shifted_size = 1 << (int) i_type; /* B -> 1, H -> 10, S -> 100, D -> 1000 */

      if (i_index >= l_elementSizeBits) {
        /* the index must be within bounds */
        fprintf(stderr, "libxsmm_aarch64_instruction_sve_compute: (instr: 0x%08x) index %d is too large for type %d, max allowed: %d!\n", i_vec_instr, i_index, (int)i_type, l_elementSizeBits);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }

      code[code_head] |= (unsigned int)((l_shifted_size >> 2) << 22); /* tszh in ARM docs */
      code[code_head] |= (unsigned int)((l_shifted_size & 0x3) << 19); /* tszl in ARM docs */
      code[code_head] |= (unsigned int)(l_index << 16); /* immediate value */
      /* set the highest bit for double at bit-index 22, if required */
      if (l_index >= 32 && i_type == LIBXSMM_AARCH64_SVE_TYPE_D) {
        code[code_head] |= (1 << 22);
      }
    } else {
      if ( l_has_two_sources ) {
        /* setting Zm */
        if ( l_is_indexed ) {
          if ( i_type == LIBXSMM_AARCH64_SVE_TYPE_S ) {
            code[code_head] |= (unsigned int)((0x7 & l_vec_reg_src_1) << 16);
            code[code_head] |= (unsigned int)((0x3 & i_index) << 19);
          } else if ( i_type == LIBXSMM_AARCH64_SVE_TYPE_D ) {
            code[code_head] |= (unsigned int)((0xf & l_vec_reg_src_1) << 16);
            code[code_head] |= (unsigned int)((0x1 & i_index) << 20);
          } /* else todo: half-type is missing */
        } else {
          code[code_head] |= (unsigned int)((0x1f & l_vec_reg_src_1) << 16);
        }
      }
      else if ( l_has_immediate ) {
        code[code_head] = (code[code_head] & 0xffffe01f) | (unsigned int)(i_index << 5);
      }
    }
    if ( l_is_type_specific ) {
      /* setting type */
      code[code_head] |= (unsigned int)((0x3 & i_type) << 22);
    }
    if ( l_is_predicated ) {
      /* setting p reg */
      code[code_head] |= (unsigned int)((0x7 & i_pred_reg) << 10);
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_compute: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_sve_pcompute( libxsmm_generated_code*           io_generated_code,
                                               const unsigned int                i_pred_instr,
                                               const unsigned int                i_pred_reg,
                                               const unsigned int                i_gp_reg_src_0,
                                               const libxsmm_aarch64_gp_width    i_gp_width,
                                               const unsigned int                i_gp_reg_src_1,
                                               const libxsmm_aarch64_sve_pattern i_pattern,
                                               const libxsmm_aarch64_sve_type    i_type ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE128 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_sve_pcompute: at least ARM SVE128 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_pred_instr ) {
    case LIBXSMM_AARCH64_INSTR_SVE_PTRUE:
      break;
    case LIBXSMM_AARCH64_INSTR_SVE_WHILELT:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_sve_pcompute: unexpected instruction number: %u\n", i_pred_instr);
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
    code[code_head]  = (unsigned int)(0xffffff00 & i_pred_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)(0xf & i_pred_reg);
    if ( (i_pred_instr & 0x3) == 0x1 ) {
      /* setting pattern */
      code[code_head] |= (unsigned int)((0x1f & i_pattern) << 5);
    }
    else if ( (i_pred_instr & 0x3) == 0x3 ) {
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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_move( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_move_instr,
                                           const unsigned int      i_gp_reg_addr,
                                           const unsigned int      i_gp_reg_offset,
                                           const int               i_offset,
                                           const unsigned int      i_gp_reg_dst ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_move_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_LDR_R:
    case LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF:
    case LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST:
    case LIBXSMM_AARCH64_INSTR_GP_LDR_I_PRE:
    case LIBXSMM_AARCH64_INSTR_GP_LDRH_I_OFF:
    case LIBXSMM_AARCH64_INSTR_GP_LDRH_I_POST:
    case LIBXSMM_AARCH64_INSTR_GP_LDRH_I_PRE:
    case LIBXSMM_AARCH64_INSTR_GP_STR_R:
    case LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF:
    case LIBXSMM_AARCH64_INSTR_GP_STR_I_POST:
    case LIBXSMM_AARCH64_INSTR_GP_STR_I_PRE:
    case LIBXSMM_AARCH64_INSTR_GP_STRH_I_OFF:
    case LIBXSMM_AARCH64_INSTR_GP_STRH_I_POST:
    case LIBXSMM_AARCH64_INSTR_GP_STRH_I_PRE:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: unexpected instruction number: %u\n", i_move_instr);
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
        int l_imm;
        /* adjust offset based on vector width */
        if ( (0x20 & i_gp_reg_dst) == 0x20 ) {
          l_imm = i_offset/8;
        } else {
          l_imm = i_offset/4;
        }
        if ( (l_imm > 0x0fff) || (i_offset < 0) ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: offset for unsigned offnset addressing mode out of range: %i, %i!\n", l_imm, i_offset);
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }
        code[code_head] |= (unsigned int)((0x00000fff & l_imm) << 10);
      } else {
        if ( (i_offset < -256) || (i_offset > 255) ) {
          fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: offset for per-index/post-index addressing mode out of range: %i!\n", i_offset);
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }
        code[code_head] |= (unsigned int)((0x000001ff & i_offset) << 12);
      }
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_pair_move( libxsmm_generated_code*           io_generated_code,
                                                const unsigned int                i_move_instr,
                                                const unsigned int                i_gp_reg_addr,
                                                const int                         i_offset,
                                                const unsigned int                i_gp_reg_0,
                                                const unsigned int                i_gp_reg_1 ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_pair_move: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
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
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;
    unsigned char l_opc = 0x0;
    signed char l_imm = 0x0;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    if ( (0x20 & i_gp_reg_0) == 0x20 ) {
      l_opc = 0x1;
      l_imm = (char)(i_offset/8);
    } else {
      l_opc = 0x0;
      l_imm = (char)(i_offset/4);
    }

    if ( (l_imm < -64) || (l_imm > 63) ) {
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_move: offset out of range: %i!\n", i_offset);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_move_imm16( libxsmm_generated_code* io_generated_code,
                                                 const unsigned int      i_alu_instr,
                                                 const unsigned int      i_gp_reg_dst,
                                                 const unsigned char     i_shift,
                                                 const unsigned int      i_imm16 ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_move_imm16: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_alu_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_MOVZ:
    case LIBXSMM_AARCH64_INSTR_GP_MOVN:
    case LIBXSMM_AARCH64_INSTR_GP_MOVK:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_move_imm16: unexpected instruction number: %u\n", i_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( ((i_gp_reg_dst < LIBXSMM_AARCH64_GP_REG_X0) && (i_shift > 1)) || (i_shift > 3) ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_move_imm16: unexpected shift: %u %u %u\n", i_alu_instr, i_gp_reg_dst, (unsigned int)i_shift);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned char l_hw = (unsigned char)(i_gp_reg_dst < LIBXSMM_AARCH64_GP_REG_X0)
      ? (0x1 & i_shift) : (0x3 & i_shift); /* computing hw */
    unsigned int code_head = io_generated_code->code_size/4;
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_set_imm64( libxsmm_generated_code*  io_generated_code,
                                                const unsigned int       i_gp_reg_dst,
                                                const unsigned long long i_imm64 ) {
  if ( i_imm64 <=         0xffff ) {
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
                                                    const unsigned int      i_gp_reg_src,
                                                    const unsigned int      i_gp_reg_dst,
                                                    const unsigned int      i_imm12,
                                                    const unsigned char     i_imm12_lsl12 ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm12: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_alu_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_ADD_I:
    case LIBXSMM_AARCH64_INSTR_GP_SUB_I:
#if 0
    case LIBXSMM_AARCH64_INSTR_GP_ORR_I:
    case LIBXSMM_AARCH64_INSTR_GP_AND_I:
    case LIBXSMM_AARCH64_INSTR_GP_EOR_I:
    case LIBXSMM_AARCH64_INSTR_GP_LSL_I:
    case LIBXSMM_AARCH64_INSTR_GP_LSR_I:
    case LIBXSMM_AARCH64_INSTR_GP_ASR_I:
#endif
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm12: unexpected instruction number: %u\n", i_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  /* check for imm being in range */
  if ( (i_imm12 > 0xfff) || (i_imm12_lsl12 > 1) ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm12: unexpected imm/shift: %u %u %u\n", i_alu_instr, (unsigned int)i_imm12, (unsigned int)i_imm12_lsl12);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* check that all regs are either 32 or 64 bit */
  if ( ((i_gp_reg_src > 31) && ( i_gp_reg_dst > 31 )) ) {
    /* nothing */
  } else if ( ((i_gp_reg_src < 32) && ( i_gp_reg_dst < 32 )) ) {
    /* nothing */
  } else {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm12: all regsiters need to be either 32 or 64bit; instr: %u\n", i_alu_instr);
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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_imm24( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_alu_instr,
                                                    const unsigned int      i_gp_reg_src,
                                                    const unsigned int      i_gp_reg_dst,
                                                    const unsigned int      i_imm24 ) {
  if ( i_imm24 > 0xffffff ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm24: unexpected imm/shift: %u %u\n", i_alu_instr, i_imm24);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( i_imm24 <= 0xfff ) {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, i_alu_instr, i_gp_reg_src, i_gp_reg_dst,
                                                   (unsigned short)(0xfff & i_imm24), 0);
  } else if ( (i_imm24 & 0xfff) == 0 ) {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, i_alu_instr, i_gp_reg_src, i_gp_reg_dst,
                                                   (unsigned short)(0xfff & (i_imm24 >> 12)), 1);
  } else {
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, i_alu_instr, i_gp_reg_src, i_gp_reg_dst,
                                                   (unsigned short)(0xfff & i_imm24), 0);
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, i_alu_instr, i_gp_reg_dst, i_gp_reg_dst,
                                                   (unsigned short)(0xfff & (i_imm24 >> 12)), 1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_shifted_reg( libxsmm_generated_code*         io_generated_code,
                                                          const unsigned int              i_alu_instr,
                                                          const unsigned int              i_gp_reg_src_0,
                                                          const unsigned int              i_gp_reg_src_1,
                                                          const unsigned int              i_gp_reg_dst,
                                                          const unsigned int              i_imm6,
                                                          const libxsmm_aarch64_shiftmode i_shift_dir ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_shifted_reg: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_alu_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_ORR_SR:
    case LIBXSMM_AARCH64_INSTR_GP_AND_SR:
    case LIBXSMM_AARCH64_INSTR_GP_EOR_SR:
    case LIBXSMM_AARCH64_INSTR_GP_LSL_SR:
    case LIBXSMM_AARCH64_INSTR_GP_LSR_SR:
    case LIBXSMM_AARCH64_INSTR_GP_ASR_SR:
    case LIBXSMM_AARCH64_INSTR_GP_ADD_SR:
    case LIBXSMM_AARCH64_INSTR_GP_SUB_SR:
    case LIBXSMM_AARCH64_INSTR_GP_MUL:
    case LIBXSMM_AARCH64_INSTR_GP_UDIV:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_shifted_reg: unexpected instruction number: %u\n", i_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  /* check for imm being in range */
  if ( (i_imm6 > 0x3f) && ((i_alu_instr & 0x4) == 0x4) ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_shifted_reg: unexpected imm: %u %u\n", i_alu_instr, (unsigned int)i_imm6);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* check that all regs are either 32 or 64 bit */
  if ( (i_gp_reg_src_0 > 31) && (i_gp_reg_src_1 > 31) && ( i_gp_reg_dst > 31 ) ) {
    /* nothing */
  } else if ( (i_gp_reg_src_0 < 32) && (i_gp_reg_src_1 < 32) && ( i_gp_reg_dst < 32 ) ) {
    /* nothing */
  } else {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_shifted_reg: all regsiters need to be either 32 or 64bit; instr: %u\n", i_alu_instr);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned char l_imm = (unsigned char)(i_gp_reg_dst < LIBXSMM_AARCH64_GP_REG_X0)
      ? (0x1f & i_imm6) : (0x3f & i_imm6); /* computing hw */
    unsigned int* code     = (unsigned int *)io_generated_code->generated_code;
    unsigned int code_head = io_generated_code->code_size / 4;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size - io_generated_code->code_size < 4 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

     /* fix bits */
    code[code_head]  = (unsigned int)(0xffffff00 & i_alu_instr);
    /* setting Rd */
    code[code_head] |= (unsigned int)(0x1f & i_gp_reg_dst);
    /* setting Rn */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_src_0) << 5);
    /* setting Rm */
    code[code_head] |= (unsigned int)((0x1f & i_gp_reg_src_1) << 16);
    /* setting sf */
    code[code_head] |= (unsigned int)((0x20 & i_gp_reg_dst) << 26);
    /* setting imm6 */
    if ( (i_alu_instr & 0x4) == 0x4 ) {
      code[code_head] |= (unsigned int)((0x3f & l_imm) << 10);
    }
    /* setting sh */
    if ( (i_alu_instr & 0x10) == 0x0 ) {
      code[code_head] |= (unsigned int)((0x3  & i_shift_dir) << 22);
    }

    /* advance code head */
    io_generated_code->code_size += 4;
  } else {
    /* assembly not supported right now */
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_shifted_reg: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_alu_compute_imm64( libxsmm_generated_code*         io_generated_code,
                                                    const unsigned int              i_alu_meta_instr,
                                                    const unsigned int              i_gp_reg_src,
                                                    const unsigned int              i_gp_reg_tmp,
                                                    const unsigned int              i_gp_reg_dst,
                                                    const unsigned long long        i_imm64 ) {
  unsigned int l_alu_instr = LIBXSMM_AARCH64_INSTR_UNDEF;

  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm64: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_alu_meta_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_META_ADD:
    case LIBXSMM_AARCH64_INSTR_GP_META_SUB:
      break;
    default:
      fprintf(stderr, "libxsmm_aarch64_instruction_alu_compute_imm64: unexpected instruction number: %u\n", i_alu_meta_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  /* check for imm being in range */
  if ( i_imm64 <= 0xffffff ) {
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
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
    }
    l_imm24 = (unsigned int)(i_imm64 & 0xffffff);

    libxsmm_aarch64_instruction_alu_compute_imm24( io_generated_code, l_alu_instr,
                                                   i_gp_reg_src, i_gp_reg_dst, l_imm24 );
  } else {
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
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
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
void libxsmm_aarch64_instruction_cond_jump_back_to_label( libxsmm_generated_code*     io_generated_code,
                                                          const unsigned int          i_jmp_instr,
                                                          const unsigned int          i_gp_reg_cmp,
                                                          libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_cond_jump_back_to_label: at least ARM V81 needs to be specified as target arch!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  switch ( i_jmp_instr ) {
    case LIBXSMM_AARCH64_INSTR_GP_CBNZ:
    case LIBXSMM_AARCH64_INSTR_GP_CBZ:
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

LIBXSMM_API_INTERN
void libxsmm_aarch64_instruction_register_jump_label( libxsmm_generated_code*     io_generated_code,
                                                      const unsigned int          i_label_no,
                                                      libxsmm_jump_label_tracker* io_jump_label_tracker ) {
  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_register_jump_label: at least ARM V81 needs to be specified as target arch!\n");
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
void libxsmm_aarch64_instruction_cond_jump_to_label( libxsmm_generated_code*     io_generated_code,
                                                     const unsigned int          i_jmp_instr,
                                                     const unsigned int          i_gp_reg_cmp,
                                                     const unsigned int          i_label_no,
                                                     libxsmm_jump_label_tracker* io_jump_label_tracker ) {
  unsigned int l_pos;

  if ( io_generated_code->arch < LIBXSMM_AARCH64_V81 ) {
    fprintf(stderr, "libxsmm_aarch64_instruction_cond_jump_to_label: at least ARM V81 needs to be specified as target arch!\n");
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
    case LIBXSMM_AARCH64_INSTR_GP_CBNZ:
    case LIBXSMM_AARCH64_INSTR_GP_CBZ:
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
