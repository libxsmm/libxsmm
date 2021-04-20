/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Aleander Breuer (Jena Univ.), Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <generator_aarch64_instructions.h>

void reset_code_buffer( libxsmm_generated_code* mycode, char* test_name ) {
  printf("Reset code buffer for testing: %s\n", test_name );
  mycode->code_size = 0;
  mycode->code_type = 2;
  mycode->last_error = 0;
  mycode->sf_size = 0;
  memset( (unsigned char*)mycode->generated_code, 0, mycode->buffer_size );
}

void dump_code_buffer( libxsmm_generated_code* mycode, char* test_name ) {
  FILE *fp;
  char filename[255];

  memset( filename, 0, 255);
  strcat( filename, test_name );
  strcat( filename, ".bin" );

  fp = fopen( filename, "wb" );
  if (fp == NULL) {
    printf("Error opening binary dumping file!\n");
    exit(1);
  }
  fwrite(mycode->generated_code, sizeof(unsigned char), mycode->code_size, fp);
  fclose(fp);
}

void test_asimd_move( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char b;
  unsigned char o;
  int simd[5] = {LIBXSMM_AARCH64_ASIMD_WIDTH_B, LIBXSMM_AARCH64_ASIMD_WIDTH_H, LIBXSMM_AARCH64_ASIMD_WIDTH_S, LIBXSMM_AARCH64_ASIMD_WIDTH_D, LIBXSMM_AARCH64_ASIMD_WIDTH_Q};
  unsigned char v;
  unsigned char w;
  short offset = 64;

  reset_code_buffer( mycode, test_name );

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

  dump_code_buffer( mycode, test_name );
}

void test_asimd_struct_move( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char b;
  unsigned char o;
  int simd[8] = { LIBXSMM_AARCH64_ASIMD_SCRUCTTYPE_8B, LIBXSMM_AARCH64_ASIMD_SCRUCTTYPE_16B, LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_4H, LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_8H,
                  LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_2S,  LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_4S, LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_1D, LIBXSMM_AARCH64_ASIMD_STRUCTTYPE_2D };
  unsigned char v;
  unsigned char w;

  reset_code_buffer( mycode, test_name );

  if ( (instr == LIBXSMM_AARCH64_INSTR_ASIMD_LD1R) ) {
    for (b = 32; b < 64; ++b ) {
      for (v = 0; v < 32; ++v ) {
        for (w = 0; w < 8; ++w ) {
          libxsmm_aarch64_instruction_asimd_struct_move( mycode, instr, b, LIBXSMM_AARCH64_GP_REG_UNDEF, v, simd[w] );
        }
      }
    }
  } else {
    for (b = 32; b < 64; ++b ) {
      for (o = 32; o < 64; ++o ) {
        for (v = 0; v < 32; ++v ) {
          for (w = 0; w < 8; ++w ) {
            libxsmm_aarch64_instruction_asimd_struct_move( mycode, instr, b, o, v, simd[w] );
          }
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_asimd_pair_move( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char b;
  int simd[3] = {LIBXSMM_AARCH64_ASIMD_WIDTH_S, LIBXSMM_AARCH64_ASIMD_WIDTH_D, LIBXSMM_AARCH64_ASIMD_WIDTH_Q};
  unsigned char w;
  unsigned char s;
  unsigned char t;
  short offset = 64;

  reset_code_buffer( mycode, test_name );

  for (b = 32; b < 64; ++b ) {
    for (s = 0; s < 32; ++s ) {
      for (t = 0; t < 32; ++t ) {
        for (w = 0; w < 3; ++w ) {
          libxsmm_aarch64_instruction_asimd_pair_move( mycode, instr, b, offset, s, t, simd[w] );
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_asimd_compute( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned char has_index ) {
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

  reset_code_buffer( mycode, test_name );

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

  dump_code_buffer( mycode, test_name );
}

void test_sve_move( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, short offset ) {
  unsigned char b;
  unsigned char o;
  unsigned char v;
  unsigned char p;

  reset_code_buffer( mycode, test_name );

  if (offset == 0) {
    for (b = 32; b < 64; ++b ) {
      for (o = 32; o < 64; ++o ) {
        for (v = 0; v < 32; ++v ) {
          for (p = 0; p < 8; ++p ) {
            libxsmm_aarch64_instruction_sve_move( mycode, instr, b, o, 0, v, p );
          }
        }
      }
    }
  } else {
    for (b = 32; b < 64; ++b ) {
      for (v = 0; v < 32; ++v ) {
        for (p = 0; p < 8; ++p ) {
          libxsmm_aarch64_instruction_sve_move( mycode, instr, b, LIBXSMM_AARCH64_GP_REG_UNDEF, offset, v, p );
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_sve_compute( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned char has_pred ) {
  unsigned char d;
  int tuple[3] = {LIBXSMM_AARCH64_SVE_TYPE_H, LIBXSMM_AARCH64_SVE_TYPE_S, LIBXSMM_AARCH64_SVE_TYPE_D};
  unsigned char s;
  unsigned char t;
  unsigned char w;
  unsigned char p;

  reset_code_buffer( mycode, test_name );

  for (d = 0; d < 32; ++d ) {
    for (s = 0; s < 32; ++s ) {
      for (t = 0; t < 32; ++t ) {
        for (w = 0; w < 3; ++w ) {
          if ( has_pred == 0 ) {
            libxsmm_aarch64_instruction_sve_compute( mycode, instr, s, t, 0, d, 0, tuple[w] );
          } else {
            for (p = 0; p < 8; ++p) {
              libxsmm_aarch64_instruction_sve_compute( mycode, instr, s, t, 0, d, p, tuple[w] );
            }
          }
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_sve_pcompute( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char d;
  int tuple[3] = {LIBXSMM_AARCH64_SVE_PATTERN_ALL, LIBXSMM_AARCH64_SVE_PATTERN_VL128, LIBXSMM_AARCH64_SVE_PATTERN_VL1};
  int size[4] = {LIBXSMM_AARCH64_SVE_TYPE_B, LIBXSMM_AARCH64_SVE_TYPE_H, LIBXSMM_AARCH64_SVE_TYPE_S, LIBXSMM_AARCH64_SVE_TYPE_D};
  unsigned char t;
  unsigned char p;

  reset_code_buffer( mycode, test_name );

  for (d = 0; d < 8; ++d ) {
    for (t = 0; t < 4; ++t ) {
      for (p = 0; p < 3; ++p ) {
        libxsmm_aarch64_instruction_sve_pcompute( mycode, instr, d, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, 0, LIBXSMM_AARCH64_ASIMD_REG_UNDEF, tuple[p], size[t] );
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_alu_move( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char b;
  unsigned char o;
  unsigned char d;
  short offset = 64;

  reset_code_buffer( mycode, test_name );

  if ( (instr == LIBXSMM_AARCH64_INSTR_GP_LDR_R) || (instr == LIBXSMM_AARCH64_INSTR_GP_STR_R) ) {
    for (b = 32; b < 64; ++b ) {
      for (o = 0; o < 64; ++o ) {
        for (d = 0; d < 64; ++d ) {
          libxsmm_aarch64_instruction_alu_move( mycode, instr, b, o, 0, d );
        }
      }
    }
  } else {
    for (b = 32; b < 64; ++b ) {
      for (d = 0; d < 64; ++d ) {
        libxsmm_aarch64_instruction_alu_move( mycode, instr, b, 0, offset, d );
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_alu_pair_move( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char b;
  unsigned char s;
  unsigned char t;
  short offset = 64;

  reset_code_buffer( mycode, test_name );

  for (b = 32; b < 64; ++b ) {
    for (s = 0; s < 64; ++s ) {
      unsigned t_start = ( s < 32 ) ? 0 : 32;
      for (t = t_start; t < t_start + 32; ++t ) {
        libxsmm_aarch64_instruction_alu_pair_move( mycode, instr, b, offset, s, t );
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_gpr_alu_move_imm16( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char d;
  unsigned char shift[4] = {0,1,2,3};
  unsigned short imm[4] = {0x44,0x33,0x22,0x11};
  unsigned char i;

  reset_code_buffer( mycode, test_name );

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

  dump_code_buffer( mycode, test_name );
}

void test_gpr_alu_compute_imm12( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char d;
  unsigned char s;
  unsigned char shift[2] = {0,1};
  unsigned short imm = 0x443;
  unsigned char i;

  reset_code_buffer( mycode, test_name );

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

  dump_code_buffer( mycode, test_name );
}

void test_gpr_alu_compute_shifted_reg( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char d;
  unsigned char s;
  unsigned char t;
  libxsmm_aarch64_shiftmode shift[3] = {LIBXSMM_AARCH64_SHIFTMODE_LSL, LIBXSMM_AARCH64_SHIFTMODE_LSR, LIBXSMM_AARCH64_SHIFTMODE_ASR};
  unsigned short imm = 0x14;
  unsigned char i;

  reset_code_buffer( mycode, test_name );

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

  dump_code_buffer( mycode, test_name );
}

int main( /*int argc, char* argv[]*/ ) {
  unsigned char* codebuffer = (unsigned char*)malloc( 8388608*sizeof(unsigned char) );
  libxsmm_generated_code mycode;

  /* init generated code object */
  mycode.generated_code = codebuffer;
  mycode.buffer_size = 8388608;
  mycode.arch = LIBXSMM_AARCH64_V81;

  /* testing asimd ldr/str instructions */
  test_asimd_move( "asimd_mov_LDR_R", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R );
  test_asimd_move( "asimd_mov_LDR_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_OFF );
  test_asimd_move( "asimd_mov_LDR_I_POST", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_POST );
  test_asimd_move( "asimd_mov_LDR_I_PRE", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDR_I_PRE );
  test_asimd_move( "asimd_mov_LDR_STR_R", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STR_R );
  test_asimd_move( "asimd_mov_STR_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_OFF );
  test_asimd_move( "asimd_mov_STR_I_POST", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_POST );
  test_asimd_move( "asimd_mov_STR_I_PRE", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STR_I_PRE );

  test_asimd_struct_move( "asimd_mov_LD1R", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R );
  test_asimd_struct_move( "asimd_mov_LD1R_R_POST", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LD1R_R_POST );

  test_asimd_pair_move( "asimd_movp_LDP_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_OFF );
  test_asimd_pair_move( "asimd_movp_LDP_I_POST", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_POST );
  test_asimd_pair_move( "asimd_movp_LDP_I_PRE", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDP_I_PRE );
  test_asimd_pair_move( "asimd_movp_LDNP_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_LDNP_I_OFF );
  test_asimd_pair_move( "asimd_movp_STP_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_OFF );
  test_asimd_pair_move( "asimd_movp_STP_I_POST", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_POST );
  test_asimd_pair_move( "asimd_movp_STP_I_PRE", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STP_I_PRE );
  test_asimd_pair_move( "asimd_movp_STNP_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_STNP_I_OFF );

  /* test SIMD compute instructions */
  test_asimd_compute( "asimd_comp_FMLA_V",   &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_V,   0 );
  test_asimd_compute( "asimd_comp_FMLA_E_V", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_V, 1 );
  test_asimd_compute( "asimd_comp_FMLA_E_S", &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_FMLA_E_S, 1 );
  test_asimd_compute( "asimd_comp_EOR_V",    &mycode, LIBXSMM_AARCH64_INSTR_ASIMD_EOR_V,    0 );

  mycode.arch = LIBXSMM_AARCH64_A64FX;
  /* testing asimd ldr/str instructions */
  test_sve_move( "sve_mov_LDR_P_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_SVE_LDR_P_I_OFF, 128 );
  test_sve_move( "sve_mov_LDR_Z_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF, 128 );
  test_sve_move( "sve_mov_STR_P_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_SVE_STR_P_I_OFF, 128 );
  test_sve_move( "sve_mov_STR_Z_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF, 128 );
  test_sve_move( "sve_mov_LD1D_SR", &mycode, LIBXSMM_AARCH64_INSTR_SVE_LD1D_SR, 0 );
  test_sve_move( "sve_mov_LD1D_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_SVE_LD1D_I_OFF, 7 );
  test_sve_move( "sve_mov_LD1W_SR", &mycode, LIBXSMM_AARCH64_INSTR_SVE_LD1W_SR, 0 );
  test_sve_move( "sve_mov_LD1W_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_SVE_LD1W_I_OFF, 7 );
  test_sve_move( "sve_mov_ST1D_SR", &mycode, LIBXSMM_AARCH64_INSTR_SVE_ST1D_SR, 0 );
  test_sve_move( "sve_mov_ST1D_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_SVE_ST1D_I_OFF, 7 );
  test_sve_move( "sve_mov_ST1W_SR", &mycode, LIBXSMM_AARCH64_INSTR_SVE_ST1W_SR, 0 );
  test_sve_move( "sve_mov_ST1W_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_SVE_ST1W_I_OFF, 7 );
  test_sve_move( "sve_mov_LD1RD_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_SVE_LD1RD_I_OFF, 64 );
  test_sve_move( "sve_mov_LD1RW_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_SVE_LD1RW_I_OFF, 64 );

  /* test SVE compute insturctions */
  test_sve_compute( "sve_comp_FMLA_V",   &mycode, LIBXSMM_AARCH64_INSTR_SVE_FMLA_V,   1 );
  test_sve_compute( "sve_comp_EOR_V",    &mycode, LIBXSMM_AARCH64_INSTR_SVE_EOR_V,    0 );
  test_sve_pcompute( "sve_pcomp_PTRUE",    &mycode, LIBXSMM_AARCH64_INSTR_SVE_PTRUE );
  mycode.arch = LIBXSMM_AARCH64_V81;

  /* testing asimd ldr/str instructions */
  test_alu_move( "alu_mov_LDR_R", &mycode, LIBXSMM_AARCH64_INSTR_GP_LDR_R );
  test_alu_move( "alu_mov_LDR_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF );
  test_alu_move( "alu_mov_LDR_I_POST", &mycode, LIBXSMM_AARCH64_INSTR_GP_LDR_I_POST );
  test_alu_move( "alu_mov_LDR_I_PRE", &mycode, LIBXSMM_AARCH64_INSTR_GP_LDR_I_PRE );
  test_alu_move( "alu_mov_STR_R", &mycode, LIBXSMM_AARCH64_INSTR_GP_STR_R );
  test_alu_move( "alu_mov_STR_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF );
  test_alu_move( "alu_mov_STR_I_POST", &mycode, LIBXSMM_AARCH64_INSTR_GP_STR_I_POST );
  test_alu_move( "alu_mov_STR_I_PRE", &mycode, LIBXSMM_AARCH64_INSTR_GP_STR_I_PRE );

  test_alu_pair_move( "alu_movp_LDP_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_GP_LDP_I_OFF );
  test_alu_pair_move( "alu_movp_LDP_I_POST", &mycode, LIBXSMM_AARCH64_INSTR_GP_LDP_I_POST );
  test_alu_pair_move( "alu_movp_LDP_I_PRE", &mycode, LIBXSMM_AARCH64_INSTR_GP_LDP_I_PRE );
  test_alu_pair_move( "alu_movp_LDNP_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_GP_LDNP_I_OFF );
  test_alu_pair_move( "alu_movp_STP_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF );
  test_alu_pair_move( "alu_movp_STP_I_POST", &mycode, LIBXSMM_AARCH64_INSTR_GP_STP_I_POST );
  test_alu_pair_move( "alu_movp_STP_I_PRE", &mycode, LIBXSMM_AARCH64_INSTR_GP_STP_I_PRE );
  test_alu_pair_move( "alu_movp_STNP_I_OFF", &mycode, LIBXSMM_AARCH64_INSTR_GP_STNP_I_OFF );

  /* ALU set imm instructions */
  test_gpr_alu_move_imm16( "aarch64_mov_i16_movz", &mycode, LIBXSMM_AARCH64_INSTR_GP_MOVZ );
  test_gpr_alu_move_imm16( "aarch64_mov_i16_movn", &mycode, LIBXSMM_AARCH64_INSTR_GP_MOVN );
  test_gpr_alu_move_imm16( "aarch64_mov_i16_movk", &mycode, LIBXSMM_AARCH64_INSTR_GP_MOVK );

  /* ALU compute imm instructions */
  test_gpr_alu_compute_imm12( "aarch64_comp_i12_ADD_I", &mycode, LIBXSMM_AARCH64_INSTR_GP_ADD_I );
  test_gpr_alu_compute_imm12( "aarch64_comp_i12_SUB_I", &mycode, LIBXSMM_AARCH64_INSTR_GP_SUB_I );
  test_gpr_alu_compute_shifted_reg( "aarch64_comp_sreg_ADD_SR", &mycode, LIBXSMM_AARCH64_INSTR_GP_ADD_SR );
  test_gpr_alu_compute_shifted_reg( "aarch64_comp_sreg_SUB_SR", &mycode, LIBXSMM_AARCH64_INSTR_GP_SUB_SR );

  free( codebuffer );

  return 0;
}
