/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <generator_aarch64_instructions.h>

void test_asimd_move( libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char b;
  unsigned char o;
  int simd[5] = {LIBXSMM_AARCH64_ASIMD_WIDTH_B, LIBXSMM_AARCH64_ASIMD_WIDTH_H, LIBXSMM_AARCH64_ASIMD_WIDTH_S, LIBXSMM_AARCH64_ASIMD_WIDTH_D, LIBXSMM_AARCH64_ASIMD_WIDTH_Q};
  unsigned char v;
  unsigned char w;
  short offset = 64;

  if ( (instr == LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R) || (instr == LIBXSMM_AARCH64_INSTR_ASIMD_STR_R) ) {
    for (b = 0; b < 32; ++b ) {
      for (o = 0; o < 64; ++o ) {
        for (v = 0; v < 32; ++v ) {
          for (w = 0; w < 5; ++w ) {
            libxsmm_aarch64_instruction_asimd_move( mycode, instr, b, o, 0, v, simd[w] );
          }
        }
      }
    }
  } else {
    for (b = 0; b < 32; ++b ) {
      for (v = 0; v < 32; ++v ) {
        for (w = 0; w < 5; ++w ) {
          libxsmm_aarch64_instruction_asimd_move( mycode, instr, b, 0, offset, v, simd[w] );
        }
      }
    }
  }
}

void test_asimd_pair_move( libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char b;
  int simd[3] = {LIBXSMM_AARCH64_ASIMD_WIDTH_S, LIBXSMM_AARCH64_ASIMD_WIDTH_D, LIBXSMM_AARCH64_ASIMD_WIDTH_Q};
  unsigned char w;
  unsigned char s;
  unsigned char t;
  short offset = 64;

  for (b = 32; b < 64; ++b ) {
    for (s = 0; s < 32; ++s ) {
      for (t = 0; t < 32; ++t ) {
        for (w = 0; w < 3; ++w ) {
          libxsmm_aarch64_instruction_asimd_pair_move( mycode, instr, b, offset, s, t, simd[w] );
        }
      }
    }
  }
}

void test_asimd_compute( libxsmm_generated_code* mycode, unsigned int instr, unsigned char has_index ) {
  unsigned char d;
#if 1
  int tuple[3] = {LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2S, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_4S, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D};
#else
  int tuple[3] = {LIBXSMM_AARCH64_ASIMD_TUPLETYPE_8B, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B, LIBXSMM_AARCH64_ASIMD_TUPLETYPE_16B};
#endif
  unsigned char s;
  unsigned char t;
  unsigned char w;
  unsigned char i;

  for (d = 0; d < 32; ++d ) {
    for (s = 0; s < 32; ++s ) {
      for (t = 0; t < 32; ++t ) {
        for (w = 0; w < 3; ++w ) {
          if ( has_index == 0 ) {
            libxsmm_aarch64_instruction_asimd_compute( mycode, instr, s, t, 0, d, tuple[w] );
          } else {
            unsigned char max_idx = 4;
            if ( tuple[w] == LIBXSMM_AARCH64_ASIMD_TUPLETYPE_2D ) {
              max_idx = 2;
            }
            for ( i = 0; i < max_idx; ++i ) {
              libxsmm_aarch64_instruction_asimd_compute( mycode, instr, s, t, i, d, tuple[w] );
            }
          }
        }
      }
    }
  }
}

void test_gpr_alu_move_imm16( libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char d;
  unsigned char shift[4] = {0,1,2,3};
  unsigned short imm[4] = {0x44,0x33,0x22,0x11};
  unsigned char i;

  for (d = 0; d < 64; ++d ) {
    if ( d < 32 ) {
      for (i = 0; i < 2; ++i ) {
        libxsmm_aarch64_instruction_alu_move_imm16( mycode, instr, d, shift[i], imm[i] );
      }
    } else {
      for (i = 0; i < 4; ++i ) {
        libxsmm_aarch64_instruction_alu_move_imm16( mycode, instr, d, shift[i], imm[i] );
      }
    }
  }
}

void test_gpr_alu_compute_imm12( libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char d;
  unsigned char s;
  unsigned char shift[2] = {0,1};
  unsigned short imm = 0x443;
  unsigned char i;

  for (d = 0; d < 32; ++d ) {
    for ( s = 0; s < 32; ++s ) {
      for (i = 0; i < 2; ++i ) {
        libxsmm_aarch64_instruction_alu_compute_imm12( mycode, instr, s, d, imm, shift[i] );
      }
    }
  }
  for (d = 32; d < 64; ++d ) {
    for ( s = 32; s < 64; ++s ) {
      for (i = 0; i < 2; ++i ) {
        libxsmm_aarch64_instruction_alu_compute_imm12( mycode, instr, s, d, imm, shift[i] );
      }
    }
  }
}

void test_gpr_alu_compute_shifted_reg( libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char d;
  unsigned char s;
  unsigned char t;
  libxsmm_aarch64_shiftmode shift[3] = {LIBXSMM_AARCH64_SHIFTMODE_LSL, LIBXSMM_AARCH64_SHIFTMODE_LSR, LIBXSMM_AARCH64_SHIFTMODE_ASR};
  unsigned short imm = 0x14;
  unsigned char i;

  for (d = 0; d < 32; ++d ) {
    for ( s = 0; s < 32; ++s ) {
      for ( t = 0; t < 32; ++t ) {
        for (i = 0; i < 3;  ++i ) {
          libxsmm_aarch64_instruction_alu_compute_shifted_reg( mycode, instr, s, t, d,   0, shift[i] );
          libxsmm_aarch64_instruction_alu_compute_shifted_reg( mycode, instr, s, t, d, imm, shift[i] );
        }
      }
    }
  }
  for (d = 32; d < 64; ++d ) {
    for ( s = 32; s < 64; ++s ) {
      for ( t = 32; t < 64; ++t ) {
        for (i = 0; i < 3;  ++i ) {
          libxsmm_aarch64_instruction_alu_compute_shifted_reg( mycode, instr, s, t, d,   0, shift[i] );
          libxsmm_aarch64_instruction_alu_compute_shifted_reg( mycode, instr, s, t, d, imm, shift[i] );
        }
      }
    }
  }
}

int main( /*int argc, char* argv[]*/ ) {
  unsigned char* codebuffer = (unsigned char*)malloc( 134217728*sizeof(unsigned char) );
  libxsmm_generated_code mycode;
  FILE *fp;

  /* init generated code object */
  mycode.generated_code = codebuffer;
  mycode.buffer_size = 134217728;
  mycode.code_size = 0;
  mycode.code_type = 2;
  mycode.last_error = 0;
  mycode.arch = LIBXSMM_AARCH64_V81;
  mycode.sf_size = 0;

  /* testing asimd ldr/str instructions */
  test_asimd_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R );
  test_asimd_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF );
  test_asimd_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST );
  test_asimd_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_PRE );
  test_asimd_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STR_R );
  test_asimd_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF );
  test_asimd_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST );
  test_asimd_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_PRE );

  test_asimd_pair_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_OFF );
  test_asimd_pair_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_POST );
  test_asimd_pair_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_PRE );
  test_asimd_pair_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDNP_OFF );
  test_asimd_pair_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STP_OFF );
  test_asimd_pair_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STP_POST );
  test_asimd_pair_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STP_PRE );
  test_asimd_pair_move( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STNP_OFF );

  /* test SIMD compute instructions */
  test_asimd_compute( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,   0 );
  test_asimd_compute( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V, 1 );
  test_asimd_compute( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_S, 1 );
  test_asimd_compute( &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,    0 );

  /* ALU set imm instructions */
  test_gpr_alu_move_imm16( &mycode, LIBXSMM_AARCH64_INSTR_GP_MOVZ );
  test_gpr_alu_move_imm16( &mycode, LIBXSMM_AARCH64_INSTR_GP_MOVN );
  test_gpr_alu_move_imm16( &mycode, LIBXSMM_AARCH64_INSTR_GP_MOVK );

  /* ALU compute imm instructions */
  test_gpr_alu_compute_imm12( &mycode, LIBXSMM_AARCH64_INSTR_GP_ADD_I );
  test_gpr_alu_compute_imm12( &mycode, LIBXSMM_AARCH64_INSTR_GP_SUB_I );
  test_gpr_alu_compute_shifted_reg( &mycode, LIBXSMM_AARCH64_INSTR_GP_ADD_SR );
  test_gpr_alu_compute_shifted_reg( &mycode, LIBXSMM_AARCH64_INSTR_GP_SUB_SR );

  /* dump stream into binday file */
  fp = fopen("bytecode_aarch64.bin", "wb");
  if (fp == NULL) {
    printf("Error opening binary dumping file!\n");
    exit(1);
  }
  fwrite(codebuffer, sizeof(unsigned char), mycode.code_size, fp);
  fclose(fp);

  free( codebuffer );

  return 0;
}
