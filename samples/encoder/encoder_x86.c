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

void test_evex_load_store( libxsmm_generated_code* mycode, unsigned int instr, unsigned int load_only ) {
  unsigned int z;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 3; ++d ) {
      for (z = 0; z < 32; ++z ) {
        libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'z', z, 0, 0, 0 );
      }
      if ( load_only == 0 ) {
        for (z = 0; z < 32; ++z ) {
          libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'z', z, 0, 0, 1 );
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for (z = 0; z < 32; ++z ) {
          libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'z', z, 0, 0, 0 );
        }
        if ( load_only == 0 ) {
          for (z = 0; z < 32; ++z ) {
            libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'z', z, 0, 0, 1 );
          }
        }
      }
    }
  }
}

void test_evex_compute_3reg_general( libxsmm_generated_code* mycode, unsigned int instr, unsigned int twoops, unsigned short imm8, unsigned int max_dst ) {
  unsigned int i;
  unsigned int m;
  unsigned int init_dst = ( max_dst == 32 ) ? 0 : 1;

  for (i = 0; i < 32; ++i ) {
    for ( m = 0; m < 8; ++m ) {
      if ( twoops ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_imm8 ( mycode, instr, 'z', i, LIBXSMM_X86_VEC_REG_UNDEF, 0, m, 0, imm8 );
      } else {
        libxsmm_x86_instruction_vec_compute_3reg_mask_imm8 ( mycode, instr, 'z', i, 0, 0, m, 0, imm8 );
      }
    }
  }
  if ( !twoops ) {
    for (i = 0; i < 32; ++i ) {
      for ( m = 0; m < 8; ++m ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_imm8 ( mycode, instr, 'z', 0, i, 0, m, 0, imm8 );
      }
    }
  }
  for (i = init_dst; i < max_dst; ++i ) {
    for ( m = 0; m < 8; ++m ) {
      if ( twoops ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_imm8 ( mycode, instr, 'z', 0, LIBXSMM_X86_VEC_REG_UNDEF, i, m, 0, imm8 );
      } else {
        libxsmm_x86_instruction_vec_compute_3reg_mask_imm8 ( mycode, instr, 'z', 0, 0, i, m, 0, imm8 );
      }
    }
  }
}


void test_evex_compute_mem_2reg_general( libxsmm_generated_code* mycode, unsigned int instr, unsigned int twoops, unsigned short imm8, unsigned int max_dst ) {
  unsigned int i;
  unsigned int m;
  unsigned int b;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;
  unsigned int z;
  unsigned int init_dst = ( max_dst == 32 ) ? 0 : 1;

  for (b = 0; b < 16; ++b ) {
    for (d = 0; d < 3; ++d ) {
      for (z = init_dst; z < max_dst; ++z ) {
        for ( m = 0; m < 8; ++m ) {
          if ( twoops ) {
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 0, LIBXSMM_X86_VEC_REG_UNDEF, z, m, 0, imm8 );
          } else {
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 0, z, 0, m, 0, imm8 );
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 0, 0, z, m, 0, imm8 );
          }
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for (d = 0; d < 3; ++d ) {
        for (z = init_dst; z < max_dst; ++z ) {
          for ( m = 0; m < 8; ++m ) {
            if ( twoops ) {
              libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, i, scale, displ[d], 0, LIBXSMM_X86_VEC_REG_UNDEF, z, m, 0, imm8 );
            } else {
              libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, i, scale, displ[d], 0, z, 0, m, 0, imm8 );
              libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, i, scale, displ[d], 0, 0, z, m, 0, imm8 );
            }
          }
        }
      }
    }
  }
}

void test_evex_convert( libxsmm_generated_code* mycode, unsigned int instr, unsigned int twoops ) {
  unsigned int i;
  unsigned int imm8 = 0;

  for (i = 0; i < 32; ++i ) {
    if ( twoops ) {
      libxsmm_x86_instruction_vec_compute_convert ( mycode, mycode->arch, instr, 'z', i, LIBXSMM_X86_VEC_REG_UNDEF, 0, imm8 );
    } else {
      libxsmm_x86_instruction_vec_compute_convert ( mycode, mycode->arch, instr, 'z', i, 0, 0, imm8 );
    }
  }
  if ( !twoops ) {
    for (i = 0; i < 32; ++i ) {
      libxsmm_x86_instruction_vec_compute_convert ( mycode, mycode->arch, instr, 'z', 0, i, 0, imm8 );
    }
  }
  for (i = 0; i < 32; ++i ) {
    if ( twoops ) {
      libxsmm_x86_instruction_vec_compute_convert ( mycode, mycode->arch, instr, 'z', 0, LIBXSMM_X86_VEC_REG_UNDEF, i, imm8 );
    } else {
      libxsmm_x86_instruction_vec_compute_convert ( mycode, mycode->arch, instr, 'z', 0, 0, i, imm8 );
    }
  }
}

void test_evex_compute_reg( libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int i;

  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_reg ( mycode, mycode->arch, instr, 'z', i, 0, 0 );
  }
  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_reg ( mycode, mycode->arch, instr, 'z', 0, i, 0 );
  }
  for (i = 0; i < 32; ++i ) {
    libxsmm_x86_instruction_vec_compute_reg ( mycode, mycode->arch, instr, 'z', 0, 0, i );
  }
}

void test_evex_compute_mem( libxsmm_generated_code* mycode, unsigned int instr ) {
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
          libxsmm_x86_instruction_vec_compute_mem( mycode, mycode->arch, instr, bcst, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'z', z, 0 );
        }
        for (z = 0; z < 32; ++z ) {
          libxsmm_x86_instruction_vec_compute_mem( mycode, mycode->arch, instr, bcst, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'z', 0, z );
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for ( bcst = 0; bcst < 2; ++bcst ) {
          for (z = 0; z < 32; ++z ) {
            libxsmm_x86_instruction_vec_compute_mem( mycode, mycode->arch, instr, bcst, b, i, scale, displ[d], 'z', z, 0 );
          }
          for (z = 0; z < 32; ++z ) {
            libxsmm_x86_instruction_vec_compute_mem( mycode, mycode->arch, instr, bcst, b, i, scale, displ[d], 'z', 0, z );
          }
        }
      }
    }
  }
}

void test_vex_load_store( libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int y;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 3; ++d ) {
      for (y = 0; y < 16; ++y ) {
        libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'y', y, 0, 0, 0 );
      }
      for (y = 0; y < 16; ++y ) {
        libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'y', y, 0, 0, 1 );
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for (y = 0; y < 16; ++y ) {
          libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'y', y, 0, 0, 0 );
        }
        for (y = 0; y < 16; ++y ) {
          libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'y', y, 0, 0, 1 );
        }
      }
    }
  }
}

void test_vex_mask_load_store( libxsmm_generated_code* mycode, unsigned int arch, unsigned int instr ) {
  unsigned int y;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;
  unsigned int m;

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 3; ++d ) {
      for (m = 0; m < 16; ++m ) {
        for (y = 0; y < 16; ++y ) {
          libxsmm_x86_instruction_vec_mask_move( mycode, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'y', y, m, 0 );
        }
        for (y = 0; y < 16; ++y ) {
          libxsmm_x86_instruction_vec_mask_move( mycode, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'y', y, m, 1 );
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for (m = 0; m < 16; ++m ) {
          for (y = 0; y < 16; ++y ) {
            libxsmm_x86_instruction_vec_mask_move( mycode, instr, b, i, scale, displ[d], 'y', y, m, 0 );
          }
          for (y = 0; y < 16; ++y ) {
            libxsmm_x86_instruction_vec_mask_move( mycode, instr, b, i, scale, displ[d], 'y', y, m, 1 );
          }
        }
      }
    }
  }
}

void test_prefetch( libxsmm_generated_code* mycode, unsigned int instr ) {
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

void test_tile_move( libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int t;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for (t = 0; t < 8; ++t ) {
          libxsmm_x86_instruction_tile_move( mycode, mycode->arch, instr, b, i, scale, displ[d], t );
        }
      }
    }
  }
}

void test_tile_compute( libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int t;

  for (t = 0; t < 8; ++t ) {
    libxsmm_x86_instruction_tile_compute ( mycode, mycode->arch, instr, t, 0, 0 );
  }
  for (t = 0; t < 8; ++t ) {
    libxsmm_x86_instruction_tile_compute ( mycode, mycode->arch, instr, 0, t, 0 );
  }
  for (t = 0; t < 8; ++t ) {
    libxsmm_x86_instruction_tile_compute ( mycode, mycode->arch, instr, 0, 0, t );
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
  mycode.arch = LIBXSMM_X86_AVX512_SPR;
  mycode.sf_size = 0;

  /* testing ld/st instructions */
#if 0
  test_evex_load_store( &mycode, LIBXSMM_X86_INSTR_VMOVUPS, 0 );
  test_evex_load_store( &mycode, LIBXSMM_X86_INSTR_VPMOVUSWB, 0 );
  test_evex_load_store( &mycode, LIBXSMM_X86_INSTR_VPMOVSWB, 0 );
  test_evex_load_store( &mycode, LIBXSMM_X86_INSTR_VPMOVWB, 0 );
  test_evex_load_store( &mycode, LIBXSMM_X86_INSTR_VPBROADCASTI64X4, 1 );
#endif

#if 0
  test_vex_load_store( &mycode, LIBXSMM_X86_INSTR_VMOVUPS );
#endif

  /* testing compute reg instructions */
#if 0
  test_evex_compute_reg( &mycode, LIBXSMM_X86_INSTR_VPERMT2B );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VEXPANDPD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VEXPANDPS, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPEXPANDD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPEXPANDQ, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPEXPANDW, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPEXPANDB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VRANGEPS, 0, 0x00, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VRANGEPD, 0, 0x00, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VRANGESS, 0, 0x00, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VRANGESD, 0, 0x00, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VMULPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VADDPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPUNPCKLWD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPUNPCKHWD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPUNPCKLDQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPUNPCKLQDQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VEXTRACTI64X4, 1, 0x01, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VPERMQ_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( &mycode, LIBXSMM_X86_INSTR_VSHUFB, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
#endif

  /* testing compute mem-reg instructions */
#if 0
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VEXPANDPD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VEXPANDPS, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VPEXPANDD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VPEXPANDQ, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VPEXPANDW, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VPEXPANDB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VRANGEPS, 0, 0x00, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VRANGEPD, 0, 0x00, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VRANGESS, 0, 0x00, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VRANGESD, 0, 0x00, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VMULPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_mem_2reg_general( &mycode, LIBXSMM_X86_INSTR_VADDPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
#endif

  /* testing convert instructions */
#if 0
  test_evex_convert( &mycode, LIBXSMM_X86_INSTR_VCVTNE2PS2BF16, 0 );
  test_evex_convert( &mycode, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, 1 );
#endif

  /* testing compute mem instructions */
#if 0
  test_evex_compute_mem( &mycode, LIBXSMM_X86_INSTR_VCVTNE2PS2BF16 );
#endif

  /* testing vex mask load */
#if 0
  test_vex_mask_load_store( &mycode, LIBXSMM_X86_INSTR_VMASKMOVPD );
  test_vex_mask_load_store( &mycode, LIBXSMM_X86_INSTR_VMASKMOVPS );
#endif

  /* testing prefetches */
#if 0
  test_prefetch( &mycode, LIBXSMM_X86_INSTR_CLDEMOTE );
  test_prefetch( &mycode, LIBXSMM_X86_INSTR_CLFLUSHOPT );
#endif

  /* testing tile move */
#if 0
  test_tile_move( &mycode, LIBXSMM_X86_INSTR_TILELOADD );
  test_tile_move( &mycode, LIBXSMM_X86_INSTR_TILELOADDT1 );
  test_tile_move( &mycode, LIBXSMM_X86_INSTR_TILESTORED );
  test_tile_move( &mycode, LIBXSMM_X86_INSTR_TILEZERO );
#endif

  /* testing tile compute */
#if 0
  test_tile_compute( &mycode, LIBXSMM_X86_INSTR_TDPBSSD );
  test_tile_compute( &mycode, LIBXSMM_X86_INSTR_TDPBSUD );
  test_tile_compute( &mycode, LIBXSMM_X86_INSTR_TDPBUSD );
  test_tile_compute( &mycode, LIBXSMM_X86_INSTR_TDPBUUD );
  test_tile_compute( &mycode, LIBXSMM_X86_INSTR_TDPBF16PS );
#endif

  /* dump stream into binday file */
  fp = fopen("bytecode.bin", "wb");
  if (fp == NULL) {
    printf("Error opening binary dumping file!\n");
    exit(1);
  }
  fwrite(codebuffer, sizeof(unsigned char), mycode.code_size, fp);
  fclose(fp);

  free( codebuffer );

  return 0;
}
