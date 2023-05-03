/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Aleander Breuer (Jena Univ.), Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <generator_riscv_instructions.h>

#define INST(o) (LIBXSMM_RISCV_INSTR_GP_##o)

#define TEST_IMM12_I(o, c, i, s, d) do          \
{                                               \
  int i1=0, i2=4095, i3=4096, i4=4097;          \
  libxsmm_riscv_instruction_##o(c, i, s, i1, d);\
  libxsmm_riscv_instruction_##o(c, i, s, i2, d);\
  libxsmm_riscv_instruction_##o(c, i, s, i3, d);\
  libxsmm_riscv_instruction_##o(c, i, s, i4, d);\
}while(0)

#define TEST_IMM20_I(o, c, i, d) do               \
{                                                 \
  int i1=0, i2=(1 << 19), i3=(1<<20), i4=(1<<21); \
  libxsmm_riscv_instruction_##o(c, i, d, i1);     \
  libxsmm_riscv_instruction_##o(c, i, d, i2);     \
  libxsmm_riscv_instruction_##o(c, i, d, i3);     \
  libxsmm_riscv_instruction_##o(c, i, d, i4);     \
}while(0)

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

void test_alu_move( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char a;
  unsigned char d;

  reset_code_buffer( mycode, test_name );

    for (a = 0; a < 33; ++a ) {
      for (d = 0; d < 33; ++d ) {
        TEST_IMM12_I( alu_move, mycode, instr, a, d );
      }
    }

  dump_code_buffer( mycode, test_name );
}

void test_alu_compute( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char a;
  unsigned char b;
  unsigned char d;

  reset_code_buffer( mycode, test_name );

  for (a = 0; a < 33; ++a ) {
    for (b = 0; b < 33; ++b ) {
      for (d = 0; d < 33; ++d ) {
        libxsmm_riscv_instruction_alu_compute( mycode, instr, a, b, d );
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_alu_compute_imm12( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char a;
  unsigned char d;

  reset_code_buffer( mycode, test_name );

  for (a = 0; a < 33; ++a ) {
    for (d = 0; d < 33; ++d ) {
      TEST_IMM12_I( alu_compute, mycode, instr, a, d );
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_alu_compute_imm20( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char d;

  reset_code_buffer( mycode, test_name );

  for (d = 0; d < 33; ++d ) {
    TEST_IMM20_I( alu_compute_imm20, mycode, instr, d );
  }

  dump_code_buffer( mycode, test_name );
}

int main( /*int argc, char* argv[]*/ ) {
  unsigned char* codebuffer = (unsigned char*)malloc( 8388608*sizeof(unsigned char) );
  libxsmm_generated_code mycode;

  /* init generated code object */
  mycode.generated_code = codebuffer;
  mycode.buffer_size = 8388608;
  mycode.arch = LIBXSMM_RISCV;

  /* testing ALU ldr/str instructions */
  test_alu_move( "alu_mov_LB", &mycode, INST(LB) );
  test_alu_move( "alu_mov_LBU", &mycode, INST(LBU) );
  test_alu_move( "alu_mov_LD", &mycode, INST(LD) );
  test_alu_move( "alu_mov_LH", &mycode, INST(LH) );
  test_alu_move( "alu_mov_LHU", &mycode, INST(LHU) );
  test_alu_move( "alu_mov_LW", &mycode, INST(LW) );
  test_alu_move( "alu_mov_LWU", &mycode, INST(LWU) );
  test_alu_move( "alu_mov_SB", &mycode, INST(SB) );
  test_alu_move( "alu_mov_SD", &mycode, INST(SD) );
  test_alu_move( "alu_mov_SH", &mycode, INST(SH) );
  test_alu_move( "alu_mov_SW", &mycode, INST(SW) );

  test_alu_compute_imm20( "alu_compute_LUI", &mycode, INST(LUI) );
  test_alu_compute_imm20( "alu_compute_AUIPC", &mycode, INST(AUIPC) );

  test_alu_compute( "alu_compute_ADD", &mycode, INST(ADD) );
  test_alu_compute( "alu_compute_ADD", &mycode, INST(SUB) );
  test_alu_compute( "alu_compute_ADD", &mycode, INST(SUBW) );
  test_alu_compute( "alu_compute_ADDW", &mycode, INST(ADDW) );
  test_alu_compute( "alu_compute_AND", &mycode, INST(AND) );
  test_alu_compute( "alu_compute_OR", &mycode, INST(OR) );
  test_alu_compute( "alu_compute_XOR", &mycode, INST(XOR) );
  test_alu_compute( "alu_compute_SLL", &mycode, INST(SLL) );
  test_alu_compute( "alu_compute_SLLW", &mycode, INST(SLLW) );
  test_alu_compute( "alu_compute_SLLW", &mycode, INST(SLT) );
  test_alu_compute( "alu_compute_SLLW", &mycode, INST(SRL) );
  test_alu_compute_imm12( "alu_compute_ADDI", &mycode, INST(ADDI) );
  test_alu_compute_imm12( "alu_compute_ADDIW", &mycode, INST(ADDIW) );
  test_alu_compute_imm12( "alu_compute_ANDI", &mycode, INST(ANDI) );
  test_alu_compute_imm12( "alu_compute_ORI", &mycode, INST(ORI) );
  test_alu_compute_imm12( "alu_compute_XORI", &mycode, INST(XORI) );
  test_alu_compute_imm12( "alu_compute_SLL", &mycode, INST(SLLI) );
  test_alu_compute_imm12( "alu_compute_SLLW", &mycode, INST(SLLIW) );
  test_alu_compute_imm12( "alu_compute_SLLW", &mycode, INST(SLTI) );
  test_alu_compute_imm12( "alu_compute_SLLW", &mycode, INST(SRLI) );

  test_alu_compute_imm12( "alu_compute_CSRRC", &mycode, INST(CSRRC) );
  test_alu_compute_imm12( "alu_compute_CSRRCI", &mycode, INST(CSRRCI) );
  test_alu_compute_imm12( "alu_compute_CSRRS", &mycode, INST(CSRRS) );
  test_alu_compute_imm12( "alu_compute_CSRRSI", &mycode, INST(CSRRSI) );

  free( codebuffer );

  return 0;
}
