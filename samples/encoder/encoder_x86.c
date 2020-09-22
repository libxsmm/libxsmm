/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include <generator_x86_instructions.h>

void test_evex_load_store( libxsmm_generated_code* mycode, unsigned int arch, unsigned int instr ) {
  unsigned int z;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  unsigned int bcst;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 3; ++d ) {
      for (z = 0; z < 32; ++z ) {
        libxsmm_x86_instruction_vec_move( mycode, arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'z', z, 0, 0, 0 );
      }
      for (z = 0; z < 32; ++z ) {
        libxsmm_x86_instruction_vec_move( mycode, arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'z', z, 0, 0, 1 );
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for (z = 0; z < 32; ++z ) {
          libxsmm_x86_instruction_vec_move( mycode, arch, instr, b, i, scale, displ[d], 'z', z, 0, 0, 0 );
        }
        for (z = 0; z < 32; ++z ) {
          libxsmm_x86_instruction_vec_move( mycode, arch, instr, b, i, scale, displ[d], 'z', z, 0, 0, 1 );
        }
      }
    }
  }
}

void test_evex_convert( libxsmm_generated_code* mycode, unsigned int arch, unsigned int instr ) {
  unsigned int i;
  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_convert ( mycode, arch, instr, 'z', i, 0, 0, 0 );
  }
  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_convert ( mycode, arch, instr, 'z', 0, i, 0, 0 );
  }
  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_convert ( mycode, arch, instr, 'z', 0, 0, i, 0 );
  }
}

void test_evex_compute_reg( libxsmm_generated_code* mycode, unsigned int arch, unsigned int instr ) {

}

void test_evex_compute_mem( libxsmm_generated_code* mycode, unsigned int arch, unsigned int instr ) {
  unsigned int z;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  unsigned int bcst;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 3; ++d ) {
      for ( bcst = 0; bcst < 2; ++bcst ) {
        for (z = 0; z < 32; ++z ) {
          libxsmm_x86_instruction_vec_compute_mem( mycode, arch, instr, bcst, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'z', z, 0 );
        }
        for (z = 0; z < 32; ++z ) {
          libxsmm_x86_instruction_vec_compute_mem( mycode, arch, instr, bcst, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'z', 0, z );
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for ( bcst = 0; bcst < 2; ++bcst ) {
          for (z = 0; z < 32; ++z ) {
            libxsmm_x86_instruction_vec_compute_mem( mycode, arch, instr, bcst, b, i, scale, displ[d], 'z', z, 0 );
          }
          for (z = 0; z < 32; ++z ) {
            libxsmm_x86_instruction_vec_compute_mem( mycode, arch, instr, bcst, b, i, scale, displ[d], 'z', 0, z );
          }
        }
      }
    }
  }
}

void test_prefetch( libxsmm_generated_code* mycode, unsigned int arch, unsigned int instr ) {
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 3; ++d ) {
      libxsmm_x86_instruction_prefetch( mycode, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d] );
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        libxsmm_x86_instruction_prefetch( mycode, instr, b, i, scale, displ[d] );
      }
    }
  }
}

int main( /*int argc, char* argv[]*/ ) {
  unsigned char codebuffer[2097152];
  unsigned int arch;
  libxsmm_generated_code mycode;
  FILE *fp;

  /* init generated code object */
  mycode.generated_code = codebuffer;
  mycode.buffer_size = 2097152;
  mycode.code_size = 0;
  mycode.code_type = 2;
  mycode.last_error = 0;
  mycode.arch = arch;
  mycode.sf_size = 0;

  /* setting arch for this test */
  arch = LIBXSMM_X86_AVX512_CPX;

  /* testing ld/st instructions */
#if 1
  test_evex_load_store( &mycode, arch, LIBXSMM_X86_INSTR_VMOVUPS );
#endif

  /* testing convert instructions */
#if 0
  test_evex_convert( &mycode, arch, LIBXSMM_X86_INSTR_VCVTNE2PS2BF16 );
#endif

  /* testing compute mem instructions */
#if 0
  test_evex_compute_mem( &mycode, arch, LIBXSMM_X86_INSTR_VCVTNE2PS2BF16 );
#endif

  /* testing prefetches */
#if 0
  test_prefetch( &mycode, arch, LIBXSMM_X86_INSTR_CLDEMOTE );
#endif

  /* dump stream into binday file */
  fp = fopen("bytecode.bin", "wb");
  if (fp == NULL) {
    printf("Error opening binary dumping file!\n");
    exit(1);
  }
  fwrite(codebuffer, sizeof(unsigned char), mycode.code_size, fp);
  fclose(fp);

  return 0;
}
