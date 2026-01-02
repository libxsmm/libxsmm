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

#include <generator_rv64_instructions.h>

#define INST(o)   (LIBXSMM_RV64_INSTR_GP_##o)
#define VINST(o)  (LIBXSMM_RV64_INSTR_RVV_##o)

#define TEST_IMM12_I(o, c, i, s, d) do          \
{                                               \
  int i1=0, i2=4095, i3=4096, i4=4097;          \
  libxsmm_rv64_instruction_##o(c, i, s, d, i1);\
  libxsmm_rv64_instruction_##o(c, i, s, d, i2);\
  libxsmm_rv64_instruction_##o(c, i, s, d, i3);\
  libxsmm_rv64_instruction_##o(c, i, s, d, i4);\
}while(0)

#define TEST_IMM20_I(o, c, i, d) do               \
{                                                 \
  int i1=0, i2=(0x00ffe), i3=(1<<20), i4=(1<<21); \
  libxsmm_rv64_instruction_##o(c, i, d, i1);     \
  libxsmm_rv64_instruction_##o(c, i, d, i2);     \
  libxsmm_rv64_instruction_##o(c, i, d, i3);     \
  libxsmm_rv64_instruction_##o(c, i, d, i4);     \
}while(0)

#define TEST_IMM64_I(o, c, d) do                  \
{                                                 \
  unsigned long long i1=0, i2=(0xffe), i3=(0xfff1), i4=(0xfffff1);\
  libxsmm_rv64_instruction_##o(c, d, i1);        \
  libxsmm_rv64_instruction_##o(c, d, i2);        \
  libxsmm_rv64_instruction_##o(c, d, i3);        \
  libxsmm_rv64_instruction_##o(c, d, i4);        \
}while(0)

void reset_code_buffer( libxsmm_generated_code* mycode, char* test_name ) {
  printf("Reset code buffer for testing: %s\n", test_name );
  mycode->code_size = 0;
  mycode->code_type = 2;
  mycode->last_error = 0;
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
        libxsmm_rv64_instruction_alu_compute( mycode, instr, a, b, d );
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

void test_rvv_setvl( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {

  reset_code_buffer( mycode, test_name );

  libxsmm_rv64_instruction_rvv_setvli( mycode, LIBXSMM_RV64_GP_REG_X3,
    LIBXSMM_RV64_GP_REG_X4, LIBXSMM_RV64_SEW_B, LIBXSMM_RV64_LMUL_M1);

  dump_code_buffer( mycode, test_name );
}

void test_rvv_move( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {

  reset_code_buffer( mycode, test_name );

  libxsmm_rv64_instruction_rvv_move( mycode, instr, LIBXSMM_RV64_GP_REG_V3,
    LIBXSMM_RV64_GP_REG_V4, LIBXSMM_RV64_GP_REG_V5, 1);

  dump_code_buffer( mycode, test_name );
}

void test_rvv_compute( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {

  reset_code_buffer( mycode, test_name );

  libxsmm_rv64_instruction_rvv_compute( mycode, instr, LIBXSMM_RV64_GP_REG_V3,
    LIBXSMM_RV64_GP_REG_V4, LIBXSMM_RV64_GP_REG_V5, 0);

  dump_code_buffer( mycode, test_name );
}

void test_rvv_compute_imm( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {

  reset_code_buffer( mycode, test_name );

  libxsmm_rv64_instruction_rvv_compute( mycode, instr, LIBXSMM_RV64_GP_REG_V3,
    0x2, LIBXSMM_RV64_GP_REG_V5, 0);

  dump_code_buffer( mycode, test_name );
}

int main( /*int argc, char* argv[]*/ ) {
  unsigned char* codebuffer = (unsigned char*)malloc( 8388608*sizeof(unsigned char) );
  libxsmm_generated_code mycode;

  /* init generated code object */
  mycode.generated_code = codebuffer;
  mycode.buffer_size = 8388608;
  mycode.arch = LIBXSMM_RV64_MVL256;

  /* testing ALU ldr/str instructions */
  test_alu_move( "alu_mov_LB", &mycode, INST(LB) );
  test_alu_move( "alu_mov_SB", &mycode, INST(SB) );

  test_alu_compute( "alu_compute_ADD", &mycode, INST(ADD) );
  test_alu_compute( "alu_compute_AND", &mycode, INST(AND) );

  test_alu_compute_imm20( "alu_compute_LUI", &mycode, INST(LUI) );

  test_alu_compute_imm12( "alu_compute_ADDI", &mycode, INST(ADDI) );
  test_alu_compute_imm12( "alu_compute_ANDI", &mycode, INST(ANDI) );

  test_cond_jump( "alu_compute_BEQ", &mycode, INST(BEQ) );

  test_jump_and_link( "alu_compute_JAL", &mycode, INST(JAL) );

  test_jump_and_link_reg( "alu_compute_JALR", &mycode, INST(JALR) );

  test_alu_set_imm64( "alu_set_imm64", &mycode, INST(LW) );

  test_rvv_setvl( "setvli", &mycode, VINST(VSETVLI) );
  test_rvv_setvl( "setivli", &mycode, VINST(VSETIVLI) );
  test_rvv_setvl( "setvl", &mycode, VINST(VSETVL) );

  test_rvv_move( "vle8_v", &mycode, VINST(VLE8_V) );
  test_rvv_move( "vle16_v", &mycode, VINST(VLE16_V) );
  test_rvv_move( "vle32_v", &mycode, VINST(VLE32_V) );
  test_rvv_move( "vle32_v", &mycode, VINST(VLE64_V) );

  test_rvv_move( "vse8_v", &mycode, VINST(VSE8_V) );
  test_rvv_move( "vse16_v", &mycode, VINST(VSE16_V) );
  test_rvv_move( "vse32_v", &mycode, VINST(VSE32_V) );
  test_rvv_move( "vse32_v", &mycode, VINST(VSE64_V) );

  test_rvv_move( "vlse8_v", &mycode, VINST(VLSE8_V) );
  test_rvv_move( "vlse16_v", &mycode, VINST(VLSE16_V) );
  test_rvv_move( "vlse32_v", &mycode, VINST(VLSE32_V) );
  test_rvv_move( "vlse32_v", &mycode, VINST(VLSE64_V) );

  test_rvv_move( "vsse8_v", &mycode, VINST(VSSE8_V) );
  test_rvv_move( "vsse16_v", &mycode, VINST(VSSE16_V) );
  test_rvv_move( "vsse32_v", &mycode, VINST(VSSE32_V) );
  test_rvv_move( "vsse32_v", &mycode, VINST(VSSE64_V) );

  test_rvv_move( "vluxei8_v", &mycode, VINST(VLUXEI8_V) );
  test_rvv_move( "vluxei16_v", &mycode, VINST(VLUXEI16_V) );
  test_rvv_move( "vluxei32_v", &mycode, VINST(VLUXEI32_V) );
  test_rvv_move( "vluxei32_v", &mycode, VINST(VLUXEI64_V) );

  test_rvv_move( "vloxei8_v", &mycode, VINST(VLOXEI8_V) );
  test_rvv_move( "vloxei16_v", &mycode, VINST(VLOXEI16_V) );
  test_rvv_move( "vloxei32_v", &mycode, VINST(VLOXEI32_V) );
  test_rvv_move( "vloxei32_v", &mycode, VINST(VLOXEI64_V) );

  test_rvv_move( "vsuxei8_v", &mycode, VINST(VSUXEI8_V) );
  test_rvv_move( "vsuxei16_v", &mycode, VINST(VSUXEI16_V) );
  test_rvv_move( "vsuxei32_v", &mycode, VINST(VSUXEI32_V) );
  test_rvv_move( "vsuxei32_v", &mycode, VINST(VSUXEI64_V) );

  test_rvv_move( "vsoxei8_v", &mycode, VINST(VSOXEI8_V) );
  test_rvv_move( "vsoxei16_v", &mycode, VINST(VSOXEI16_V) );
  test_rvv_move( "vsoxei32_v", &mycode, VINST(VSOXEI32_V) );
  test_rvv_move( "vsoxei32_v", &mycode, VINST(VSOXEI64_V) );

  test_rvv_move( "vlm_v", &mycode, VINST(VLM_V) );

  test_rvv_compute( "vadd_vv", &mycode, VINST(VADD_VV) );
  test_rvv_compute( "vadd_vx", &mycode, VINST(VADD_VX) );
  test_rvv_compute( "vfadd_vf", &mycode, VINST(VFADD_VF) );

  test_rvv_compute_imm( "vadd_vi", &mycode, VINST(VADD_VI) );

  free( codebuffer );

  return 0;
}
