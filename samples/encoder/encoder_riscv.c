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
  libxsmm_riscv_instruction_##o(c, i, s, d, i1);\
  libxsmm_riscv_instruction_##o(c, i, s, d, i2);\
  libxsmm_riscv_instruction_##o(c, i, s, d, i3);\
  libxsmm_riscv_instruction_##o(c, i, s, d, i4);\
}while(0)

#define TEST_IMM20_I(o, c, i, d) do               \
{                                                 \
  int i1=0, i2=(0x00ffe), i3=(1<<20), i4=(1<<21); \
  libxsmm_riscv_instruction_##o(c, i, d, i1);     \
  libxsmm_riscv_instruction_##o(c, i, d, i2);     \
  libxsmm_riscv_instruction_##o(c, i, d, i3);     \
  libxsmm_riscv_instruction_##o(c, i, d, i4);     \
}while(0)

#define TEST_IMM64_I(o, c, d) do                  \
{                                                 \
  unsigned long long i1=0, i2=(0xffe), i3=(0xfff1), i4=(0xfffff1);\
  libxsmm_riscv_instruction_##o(c, d, i1);        \
  libxsmm_riscv_instruction_##o(c, d, i2);        \
  libxsmm_riscv_instruction_##o(c, d, i3);        \
  libxsmm_riscv_instruction_##o(c, d, i4);        \
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

void test_alu_set_imm64( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char a;
  unsigned char d;

  reset_code_buffer( mycode, test_name );

  for (d = 0; d < 2; ++d ) {
    TEST_IMM64_I( alu_set_imm64, mycode, d );
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
      TEST_IMM12_I( alu_compute_imm12, mycode, instr, a, d );
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

void test_cond_jump( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char a;
  unsigned char d;

  reset_code_buffer( mycode, test_name );

  for (a = 0; a < 33; ++a ) {
    for (d = 0; d < 33; ++d ) {
      TEST_IMM12_I( cond_jump, mycode, instr, a, d );
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_jump_and_link_reg( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char a;
  unsigned char d;

  reset_code_buffer( mycode, test_name );

  for (a = 0; a < 33; ++a ) {
    for (d = 0; d < 33; ++d ) {
      TEST_IMM12_I( jump_and_link_reg, mycode, instr, a, d );
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_jump_and_link( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned char d;

  reset_code_buffer( mycode, test_name );

  for (d = 5; d < 7; ++d ) {
    TEST_IMM20_I( jump_and_link, mycode, instr, d );
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

  test_alu_compute( "alu_compute_ADD", &mycode, INST(ADD) );
  test_alu_compute( "alu_compute_AND", &mycode, INST(AND) );

  test_alu_compute_imm20( "alu_compute_LUI", &mycode, INST(LUI) );

  test_alu_compute_imm12( "alu_compute_ADDI", &mycode, INST(ADDI) );
  test_alu_compute_imm12( "alu_compute_ANDI", &mycode, INST(ANDI) );

  test_cond_jump( "alu_compute_BEQ", &mycode, INST(BEQ) );

  test_jump_and_link( "alu_compute_JAL", &mycode, INST(JAL) );

  test_jump_and_link_reg( "alu_compute_JALR", &mycode, INST(JALR) );

  test_alu_set_imm64( "alu_set_imm64", &mycode, INST(LW) );

  free( codebuffer );

  return 0;
}
