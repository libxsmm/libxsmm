/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <generator_x86_instructions.h>

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

void test_evex_load_store( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int load_store_cntl ) {
  unsigned int z;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for (z = 0; z < 32; ++z ) {
      for ( d = 0; d < 3; ++d ) {
        if ( (load_store_cntl & 0x1) == 0x1 ) {
          libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'z', z, 0, 0, 0 );
        }
        if ( (load_store_cntl & 0x2) == 0x2 ) {
          libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'z', z, 0, 0, 1 );
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for (z = 0; z < 32; ++z ) {
        for ( d = 0; d < 3; ++d ) {
          if ( (load_store_cntl & 0x1) == 0x1 ) {
            libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'z', z, 0, 0, 0 );
          }
          if ( (load_store_cntl & 0x2) == 0x2 ) {
            libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'z', z, 0, 0, 1 );
          }
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_evex_gathscat( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int is_gather ) {
  unsigned int z;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;
  unsigned int k;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 32; ++i ) {
      for (z = 0; z < 32; ++z ) {
        for ( d = 0; d < 3; ++d ) {
          for ( k = 1; k < 8; ++k ) {
            if ( (is_gather == 0) || ( i != z ) ) {
              libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'z', z, k, 0, 0 );
            }
          }
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_mask_move( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int b;
  unsigned int k;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for (k = 1; k < 8; ++k ) {
      libxsmm_x86_instruction_mask_move( mycode, instr, b, k );
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_mask_move_mem( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int k;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for (k = 1; k < 8; ++k ) {
      for ( d = 0; d < 3; ++d ) {
        libxsmm_x86_instruction_mask_move_mem( mycode, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], k );
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for (k = 1; k < 8; ++k ) {
        for ( d = 0; d < 3; ++d ) {
          libxsmm_x86_instruction_mask_move_mem( mycode, instr, b, i, scale, displ[d], k );
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_evex_compute_3reg_general( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int twoops, unsigned short imm8, unsigned int max_dst ) {
  unsigned int i;
  unsigned int m;
  unsigned int init_dst = ( max_dst == 32 ) ? 0 : 1;

  reset_code_buffer( mycode, test_name );

  for (i = 0; i < 32; ++i ) {
    for ( m = 0; m < 8; ++m ) {
      if ( twoops ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, 'z', i, LIBXSMM_X86_VEC_REG_UNDEF, 0, m, 0, 0, imm8 );
      } else {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, 'z', i, 0, 0, m, 0, 0, imm8 );
      }
    }
  }
  if ( !twoops ) {
    for (i = 0; i < 32; ++i ) {
      for ( m = 0; m < 8; ++m ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, 'z', 0, i, 0, m, 0, 0, imm8 );
      }
    }
  }
  for (i = init_dst; i < max_dst; ++i ) {
    for ( m = 0; m < 8; ++m ) {
      if ( twoops ) {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, 'z', 0, LIBXSMM_X86_VEC_REG_UNDEF, i, m, 0, 0, imm8 );
      } else {
        libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, 'z', 0, 0, i, m, 0, 0, imm8 );
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_mask_compute_reg( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int twoops, unsigned short imm8 ) {
  unsigned int i;

  reset_code_buffer( mycode, test_name );

  for (i = 0; i < 8; ++i ) {
    if ( twoops ) {
      libxsmm_x86_instruction_mask_compute_reg ( mycode, instr, i, LIBXSMM_X86_VEC_REG_UNDEF, 0, imm8 );
    } else {
      libxsmm_x86_instruction_mask_compute_reg ( mycode, instr, i, 0, 0, imm8 );
    }
  }
  if ( !twoops ) {
    for (i = 0; i < 8; ++i ) {
      libxsmm_x86_instruction_mask_compute_reg ( mycode, instr, 0, i, 0, imm8 );
    }
  }
  for (i = 0; i < 8; ++i ) {
    if ( twoops ) {
      libxsmm_x86_instruction_mask_compute_reg ( mycode, instr, 0, LIBXSMM_X86_VEC_REG_UNDEF, i, imm8 );
    } else {
      libxsmm_x86_instruction_mask_compute_reg ( mycode, instr, 0, 0, i, imm8 );
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_vex_compute_3reg_general( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int twoops, unsigned short imm8, unsigned int only_xmm ) {
  unsigned int i;
  unsigned char reg = ( only_xmm != 0 ) ? 'x' : 'y';

  reset_code_buffer( mycode, test_name );

  for (i = 0; i < 16; ++i ) {
    if ( twoops ) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, reg, i, LIBXSMM_X86_VEC_REG_UNDEF, 0, 0, 0, 0, imm8 );
    } else {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, reg, i, 0, 0, 0, 0, 0, imm8 );
    }
  }
  if ( !twoops ) {
    for (i = 0; i < 16; ++i ) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, reg, 0, i, 0, 0, 0, 0, imm8 );
    }
  }
  for (i = 0; i < 16; ++i ) {
    if ( twoops ) {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, reg, 0, LIBXSMM_X86_VEC_REG_UNDEF, i, 0, 0, 0, imm8 );
    } else {
      libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, reg, 0, 0, i, 0, 0, 0, imm8 );
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_evex_compute_mem_2reg_general( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int reg_args, unsigned short imm8, unsigned int max_dst, unsigned int no_mask, unsigned int no_bcst ) {
  unsigned int i;
  unsigned int m;
  unsigned int b;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;
  unsigned int z;
  unsigned int y;
  unsigned int init_dst = ( max_dst == 8 ) ? 1 : 0;
  unsigned int mloop = ( no_mask == 0 ) ? 8 : 1;
  unsigned int bcast = (no_bcst !=0 ) ? 1 : 2;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for (z = init_dst; z < max_dst; ++z ) {
      for ( m = 0; m < mloop; ++m ) {
        for (d = 0; d < 3; ++d ) {
          for (y = 0; y < bcast; ++y ) {
            if ( reg_args == 0 ) {
              libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], y, LIBXSMM_X86_VEC_REG_UNDEF, LIBXSMM_X86_VEC_REG_UNDEF, m, 0, imm8 );
            } else if ( reg_args == 1 ) {
              libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], y, LIBXSMM_X86_VEC_REG_UNDEF, z, m, 0, imm8 );
            } else {
              libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], y, z, 0, m, 0, imm8 );
              libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], y, 0, z, m, 0, imm8 );
            }
          }
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for (z = init_dst; z < max_dst; ++z ) {
        for ( m = 0; m < mloop; ++m ) {
          for (d = 0; d < 3; ++d ) {
            for (y = 0; y < bcast; ++y ) {
              if ( reg_args == 0 ) {
                libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, i, scale, displ[d], y, LIBXSMM_X86_VEC_REG_UNDEF, LIBXSMM_X86_VEC_REG_UNDEF, m, 0, imm8 );
              } else if ( reg_args == 1 ) {
                libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, i, scale, displ[d], y, LIBXSMM_X86_VEC_REG_UNDEF, z, m, 0, imm8 );
              } else {
                libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, i, scale, displ[d], y, z, 0, m, 0, imm8 );
                libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, i, scale, displ[d], y, 0, z, m, 0, imm8 );
              }
            }
          }
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_vex_compute_mem_2reg_general( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int twoops, unsigned short imm8, unsigned int only_xmm ) {
  unsigned int i;
  unsigned int b;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;
  unsigned int z;
  unsigned char reg = ( only_xmm != 0 ) ? 'x' : 'y';

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for (z = 0; z < 16; ++z ) {
      for (d = 0; d < 3; ++d ) {
        if ( twoops ) {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, reg, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 0, LIBXSMM_X86_VEC_REG_UNDEF, z, 0, 0, imm8 );
        } else {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, reg, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 0, z, 0, 0, 0, imm8 );
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, reg, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 0, 0, z, 0, 0, imm8 );
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for (z = 0; z < 16; ++z ) {
        for (d = 0; d < 3; ++d ) {
          if ( twoops ) {
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, reg, b, i, scale, displ[d], 0, LIBXSMM_X86_VEC_REG_UNDEF, z, 0, 0, imm8 );
          } else {
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, reg, b, i, scale, displ[d], 0, z, 0, 0, 0, imm8 );
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, reg, b, i, scale, displ[d], 0, 0, z, 0, 0, imm8 );
          }
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_vex_load_store( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int load_store_cntl ) {
  unsigned int y;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 3; ++d ) {
      for (y = 0; y < 16; ++y ) {
        if ( (load_store_cntl & 0x1) == 0x1 ) {
          libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'y', y, 0, 0, 0 );
        }
        if ( (load_store_cntl & 0x2) == 0x2 ) {
          libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'y', y, 0, 0, 1 );
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for (y = 0; y < 16; ++y ) {
          if ( (load_store_cntl & 0x1) == 0x1 ) {
            libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'y', y, 0, 0, 0 );
          }
          if ( (load_store_cntl & 0x2) == 0x2 ) {
            libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'y', y, 0, 0, 1 );
          }
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_rex_vcompute_2reg_general( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned short imm8 ) {
  unsigned int i;
  unsigned char reg = 'x';

  reset_code_buffer( mycode, test_name );

  for (i = 0; i < 16; ++i ) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, reg, i, LIBXSMM_X86_VEC_REG_UNDEF, 0, 0, 0, 0, imm8 );
  }
  for (i = 0; i < 16; ++i ) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, reg, 0, LIBXSMM_X86_VEC_REG_UNDEF, i, 0, 0, 0, imm8 );
  }

  dump_code_buffer( mycode, test_name );
}

void test_rex_vcompute_1reg_general( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned short imm8 ) {
  unsigned int i;
  unsigned char reg = 'x';

  reset_code_buffer( mycode, test_name );

  for (i = 0; i < 16; ++i ) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( mycode, instr, reg, LIBXSMM_X86_VEC_REG_UNDEF, LIBXSMM_X86_VEC_REG_UNDEF, i, 0, 0, 0, imm8 );
  }

  dump_code_buffer( mycode, test_name );
}

void test_rex_vcompute_mem_1reg_general( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned short imm8 ) {
  unsigned int i;
  unsigned int b;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;
  unsigned int z;
  unsigned char reg = 'x';

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for (z = 0; z < 16; ++z ) {
      for (d = 0; d < 3; ++d ) {
        libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, reg, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 0, LIBXSMM_X86_VEC_REG_UNDEF, z, 0, 0, imm8 );
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for (z = 0; z < 16; ++z ) {
        for (d = 0; d < 3; ++d ) {
          libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, reg, b, i, scale, displ[d], 0, LIBXSMM_X86_VEC_REG_UNDEF, z, 0, 0, imm8 );
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_rex_vload_vstore( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int load_store_cntl ) {
  unsigned int y;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 3; ++d ) {
      for (y = 0; y < 16; ++y ) {
        if ( (load_store_cntl & 0x1) == 0x1 ) {
          libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'x', y, 0, 0, 0 );
        }
        if ( (load_store_cntl & 0x2) == 0x2 ) {
          libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 'x', y, 0, 0, 1 );
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for (y = 0; y < 16; ++y ) {
          if ( (load_store_cntl & 0x1) == 0x1 ) {
            libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'x', y, 0, 0, 0 );
          }
          if ( (load_store_cntl & 0x2) == 0x2 ) {
            libxsmm_x86_instruction_vec_move( mycode, mycode->arch, instr, b, i, scale, displ[d], 'x', y, 0, 0, 1 );
          }
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_vex_mask_load_store( char* test_name, libxsmm_generated_code* mycode, unsigned int is_gather, unsigned int instr ) {
  unsigned int y;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;
  unsigned int m;

  reset_code_buffer( mycode, test_name );

  if ( is_gather == 0 ) {
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
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for (m = 0; m < 16; ++m ) {
          for (y = 0; y < 16; ++y ) {
            if ( (is_gather > 0) && ( (y == i) || (y == m) || (i == m) ) ) {
              /* skip */
            } else {
              libxsmm_x86_instruction_vec_mask_move( mycode, instr, b, i, scale, displ[d], 'y', y, m, 0 );
            }
          }
          if ( is_gather == 0 ) {
            for (y = 0; y < 16; ++y ) {
              libxsmm_x86_instruction_vec_mask_move( mycode, instr, b, i, scale, displ[d], 'y', y, m, 1 );
            }
          }
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_prefetch( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  reset_code_buffer( mycode, test_name );

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

  dump_code_buffer( mycode, test_name );
}

void test_tile_move( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int t;
  unsigned int b;
  unsigned int i;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for (t = 0; t < 8; ++t ) {
          libxsmm_x86_instruction_tile_move( mycode, mycode->arch, instr, b, i, scale, displ[d], t );
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_tile_compute( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int t;

  reset_code_buffer( mycode, test_name );

  for (t = 0; t < 8; ++t ) {
    libxsmm_x86_instruction_tile_compute ( mycode, mycode->arch, instr, t, 0, 0 );
  }
  for (t = 0; t < 8; ++t ) {
    libxsmm_x86_instruction_tile_compute ( mycode, mycode->arch, instr, 0, t, 0 );
  }
  for (t = 0; t < 8; ++t ) {
    libxsmm_x86_instruction_tile_compute ( mycode, mycode->arch, instr, 0, 0, t );
  }

  dump_code_buffer( mycode, test_name );
}

void test_alu_reg( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int oneop ) {
  unsigned int t;

  reset_code_buffer( mycode, test_name );

  if ( oneop != 0 ) {
    for (t = 0; t < 16; ++t ) {
      libxsmm_x86_instruction_alu_reg ( mycode, instr, LIBXSMM_X86_GP_REG_UNDEF, t );
    }
  } else {
    for (t = 0; t < 16; ++t ) {
      libxsmm_x86_instruction_alu_reg ( mycode, instr, t, 0 );
    }
    for (t = 0; t < 16; ++t ) {
      libxsmm_x86_instruction_alu_reg ( mycode, instr, 0, t );
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_alu_mem( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int load_store_cntl ) {
  unsigned int b;
  unsigned int i;
  unsigned int d;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int r;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 3; ++d ) {
      for (r = 0; r < 16; ++r ) {
        if ( (load_store_cntl & 0x1) == 0x1 ) {
          libxsmm_x86_instruction_alu_mem( mycode, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], r, 0 );
        }
        if ( (load_store_cntl & 0x2) == 0x2 ) {
          libxsmm_x86_instruction_alu_mem( mycode, instr, b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], r, 1 );
        }
      }
    }
  }
  for (b = 0; b < 16; ++b ) {
    for (i = 0; i < 16; ++i ) {
      for ( d = 0; d < 3; ++d ) {
        for (r = 0; r < 16; ++r ) {
          if ( (load_store_cntl & 0x1) == 0x1 ) {
            libxsmm_x86_instruction_alu_mem( mycode, instr, b, i, scale, displ[d], r, 0 );
          }
          if ( (load_store_cntl & 0x2) == 0x2 ) {
            libxsmm_x86_instruction_alu_mem( mycode, instr, b, i, scale, displ[d], r, 1 );
          }
        }
      }
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_alu_imm( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int b;
  unsigned int d;
  int imm[3] = {32, 128, 2097152};

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 1; ++d ) {
      libxsmm_x86_instruction_alu_imm( mycode, instr, b, imm[d] );
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_alu_imm_i64( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int b;
  unsigned int d;
  int imm[3] = {32, 128, 2097152};

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for ( d = 0; d < 1; ++d ) {
      libxsmm_x86_instruction_alu_imm_i64( mycode, instr, b, imm[d] );
    }
  }

  dump_code_buffer( mycode, test_name );
}

void test_alu_stack( char* test_name, libxsmm_generated_code* mycode, unsigned int is_pop ) {
  unsigned int b;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    if ( is_pop != 0 ) {
      libxsmm_x86_instruction_pop_reg( mycode, b );
    } else {
      libxsmm_x86_instruction_push_reg( mycode, b );
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
  mycode.arch = LIBXSMM_X86_AVX512_GNR;

  /* testing ld/st instructions */
  test_evex_load_store( "evex_mov_VMOVAPD", &mycode, LIBXSMM_X86_INSTR_VMOVAPD, 3 );
  test_evex_load_store( "evex_mov_VMOVAPS", &mycode, LIBXSMM_X86_INSTR_VMOVAPS, 3 );
  test_evex_load_store( "evex_mov_VMOVUPD", &mycode, LIBXSMM_X86_INSTR_VMOVUPD, 3 );
  test_evex_load_store( "evex_mov_VMOVUPS", &mycode, LIBXSMM_X86_INSTR_VMOVUPS, 3 );
  test_evex_load_store( "evex_mov_VMOVSS", &mycode, LIBXSMM_X86_INSTR_VMOVSS, 3 );
  test_evex_load_store( "evex_mov_VMOVSD", &mycode, LIBXSMM_X86_INSTR_VMOVSD, 3 );
  test_evex_load_store( "evex_mov_VMOVDQA32", &mycode, LIBXSMM_X86_INSTR_VMOVDQA32, 3 );
  test_evex_load_store( "evex_mov_VMOVDQA64", &mycode, LIBXSMM_X86_INSTR_VMOVDQA64, 3 );
  test_evex_load_store( "evex_mov_VMOVDQU8", &mycode, LIBXSMM_X86_INSTR_VMOVDQU8, 3 );
  test_evex_load_store( "evex_mov_VMOVDQU16", &mycode, LIBXSMM_X86_INSTR_VMOVDQU16, 3 );
  test_evex_load_store( "evex_mov_VMOVDQU32", &mycode, LIBXSMM_X86_INSTR_VMOVDQU32, 3 );
  test_evex_load_store( "evex_mov_VMOVDQU64", &mycode, LIBXSMM_X86_INSTR_VMOVDQU64, 3 );
  test_evex_load_store( "evex_mov_VMOVAPD_LD", &mycode, LIBXSMM_X86_INSTR_VMOVAPD_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVAPS_LD", &mycode, LIBXSMM_X86_INSTR_VMOVAPS_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVUPD_LD", &mycode, LIBXSMM_X86_INSTR_VMOVUPD_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVUPS_LD", &mycode, LIBXSMM_X86_INSTR_VMOVUPS_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVSS_LD", &mycode, LIBXSMM_X86_INSTR_VMOVSS_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVSD_LD", &mycode, LIBXSMM_X86_INSTR_VMOVSD_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVDQA32_LD", &mycode, LIBXSMM_X86_INSTR_VMOVDQA32_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVDQA64_LD", &mycode, LIBXSMM_X86_INSTR_VMOVDQA64_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVDQU8_LD", &mycode, LIBXSMM_X86_INSTR_VMOVDQU8_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVDQU16_LD", &mycode, LIBXSMM_X86_INSTR_VMOVDQU16_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVDQU32_LD", &mycode, LIBXSMM_X86_INSTR_VMOVDQU32_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVDQU64_LD", &mycode, LIBXSMM_X86_INSTR_VMOVDQU64_LD, 1 );
  test_evex_load_store( "evex_mov_VMOVAPD_ST", &mycode, LIBXSMM_X86_INSTR_VMOVAPD_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVAPS_ST", &mycode, LIBXSMM_X86_INSTR_VMOVAPS_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVUPD_ST", &mycode, LIBXSMM_X86_INSTR_VMOVUPD_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVUPS_ST", &mycode, LIBXSMM_X86_INSTR_VMOVUPS_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVSS_ST", &mycode, LIBXSMM_X86_INSTR_VMOVSS_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVSD_ST", &mycode, LIBXSMM_X86_INSTR_VMOVSD_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVDQA32_ST", &mycode, LIBXSMM_X86_INSTR_VMOVDQA32_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVDQA64_ST", &mycode, LIBXSMM_X86_INSTR_VMOVDQA64_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVDQU8_ST", &mycode, LIBXSMM_X86_INSTR_VMOVDQU8_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVDQU16_ST", &mycode, LIBXSMM_X86_INSTR_VMOVDQU16_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVDQU32_ST", &mycode, LIBXSMM_X86_INSTR_VMOVDQU32_ST, 2 );
  test_evex_load_store( "evex_mov_VMOVDQU64_ST", &mycode, LIBXSMM_X86_INSTR_VMOVDQU64_ST, 2 );
  test_evex_load_store( "evex_mov_VPBROADCASTD", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTD, 1 );
  test_evex_load_store( "evex_mov_VPBROADCASTQ", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTQ, 1 );
  test_evex_load_store( "evex_mov_VPBROADCASTB", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTB, 1 );
  test_evex_load_store( "evex_mov_VPBROADCASTW", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTW, 1 );
  test_evex_load_store( "evex_mov_VBROADCASTSD", &mycode, LIBXSMM_X86_INSTR_VBROADCASTSD, 1 );
  test_evex_load_store( "evex_mov_VBROADCASTSS", &mycode, LIBXSMM_X86_INSTR_VBROADCASTSS, 1 );
  test_evex_load_store( "evex_mov_VMOVNTPD", &mycode, LIBXSMM_X86_INSTR_VMOVNTPD, 2 );
  test_evex_load_store( "evex_mov_VMOVNTPS", &mycode, LIBXSMM_X86_INSTR_VMOVNTPS, 2 );
  test_evex_load_store( "evex_mov_VMOVNTDQ", &mycode, LIBXSMM_X86_INSTR_VMOVNTDQ, 2 );
  test_evex_load_store( "evex_mov_VMOVSH_LD_MEM", &mycode, LIBXSMM_X86_INSTR_VMOVSH_LD_MEM, 1 );
  test_evex_load_store( "evex_mov_VMOVSH_ST_MEM", &mycode, LIBXSMM_X86_INSTR_VMOVSH_ST_MEM, 2 );

  /* TODO: check these for stores */
  test_evex_load_store( "evex_mov_VPMOVDW", &mycode, LIBXSMM_X86_INSTR_VPMOVDW, 1 );
  test_evex_load_store( "evex_mov_VPMOVDB", &mycode, LIBXSMM_X86_INSTR_VPMOVDB, 1 );
  test_evex_load_store( "evex_mov_VPMOVSDB", &mycode, LIBXSMM_X86_INSTR_VPMOVSDB, 1 );
  test_evex_load_store( "evex_mov_VPMOVUSDB", &mycode, LIBXSMM_X86_INSTR_VPMOVUSDB, 1 );
  test_evex_load_store( "evex_mov_VPMOVSXWD", &mycode, LIBXSMM_X86_INSTR_VPMOVSXWD, 1 );
  test_evex_load_store( "evex_mov_VPMOVZXWD", &mycode, LIBXSMM_X86_INSTR_VPMOVZXWD, 1 );
  test_evex_load_store( "evex_mov_VPMOVSXBW", &mycode, LIBXSMM_X86_INSTR_VPMOVSXBW, 1 );
  test_evex_load_store( "evex_mov_VPMOVZXBW", &mycode, LIBXSMM_X86_INSTR_VPMOVZXBW, 1 );
  test_evex_load_store( "evex_mov_VPMOVSXBD", &mycode, LIBXSMM_X86_INSTR_VPMOVSXBD, 1 );
  test_evex_load_store( "evex_mov_VPMOVZXBD", &mycode, LIBXSMM_X86_INSTR_VPMOVZXBD, 1 );
  test_evex_load_store( "evex_mov_VPMOVUSWB", &mycode, LIBXSMM_X86_INSTR_VPMOVUSWB, 1 );
  test_evex_load_store( "evex_mov_VPMOVSWB", &mycode, LIBXSMM_X86_INSTR_VPMOVSWB, 1 );
  test_evex_load_store( "evex_mov_VPMOVWB", &mycode, LIBXSMM_X86_INSTR_VPMOVWB, 1 );
  test_evex_load_store( "evex_mov_VMOVDDUP", &mycode, LIBXSMM_X86_INSTR_VMOVDDUP, 1 );
  test_evex_load_store( "evex_mov_VBROADCASTI32X4", &mycode, LIBXSMM_X86_INSTR_VBROADCASTI32X4, 1 );
  test_evex_load_store( "evex_mov_VBROADCASTI64X2", &mycode, LIBXSMM_X86_INSTR_VBROADCASTI64X2, 1 );
  test_evex_load_store( "evex_mov_VBROADCASTI32X8", &mycode, LIBXSMM_X86_INSTR_VBROADCASTI32X8, 1 );
  test_evex_load_store( "evex_mov_VBROADCASTI64X4", &mycode, LIBXSMM_X86_INSTR_VBROADCASTI64X4, 1 );

  test_evex_gathscat( "evex_gathscat_VGATHERDPS", &mycode, LIBXSMM_X86_INSTR_VGATHERDPS, 1 );
  test_evex_gathscat( "evex_gathscat_VGATHERDPD", &mycode, LIBXSMM_X86_INSTR_VGATHERDPD, 1 );
  test_evex_gathscat( "evex_gathscat_VGATHERQPS", &mycode, LIBXSMM_X86_INSTR_VGATHERQPS, 1 );
  test_evex_gathscat( "evex_gathscat_VGATHERQPD", &mycode, LIBXSMM_X86_INSTR_VGATHERQPD, 1 );
  test_evex_gathscat( "evex_gathscat_VPGATHERDD", &mycode, LIBXSMM_X86_INSTR_VPGATHERDD, 1 );
  test_evex_gathscat( "evex_gathscat_VPGATHERDQ", &mycode, LIBXSMM_X86_INSTR_VPGATHERDQ, 1 );
  test_evex_gathscat( "evex_gathscat_VPGATHERQD", &mycode, LIBXSMM_X86_INSTR_VPGATHERQD, 1 );
  test_evex_gathscat( "evex_gathscat_VPGATHERQQ", &mycode, LIBXSMM_X86_INSTR_VPGATHERQQ, 1 );
  test_evex_gathscat( "evex_gathscat_VSCATTERDPS", &mycode, LIBXSMM_X86_INSTR_VSCATTERDPS, 0 );
  test_evex_gathscat( "evex_gathscat_VSCATTERDPD", &mycode, LIBXSMM_X86_INSTR_VSCATTERDPD, 0 );
  test_evex_gathscat( "evex_gathscat_VSCATTERQPS", &mycode, LIBXSMM_X86_INSTR_VSCATTERQPS, 0 );
  test_evex_gathscat( "evex_gathscat_VSCATTERQPD", &mycode, LIBXSMM_X86_INSTR_VSCATTERQPD, 0 );
  test_evex_gathscat( "evex_gathscat_VPSCATTERDD", &mycode, LIBXSMM_X86_INSTR_VPSCATTERDD, 0 );
  test_evex_gathscat( "evex_gathscat_VPSCATTERDQ", &mycode, LIBXSMM_X86_INSTR_VPSCATTERDQ, 0 );
  test_evex_gathscat( "evex_gathscat_VPSCATTERQD", &mycode, LIBXSMM_X86_INSTR_VPSCATTERQD, 0 );
  test_evex_gathscat( "evex_gathscat_VPSCATTERQQ", &mycode, LIBXSMM_X86_INSTR_VPSCATTERQQ, 0 );

  /* testing compute reg instructions */
  test_evex_compute_3reg_general( "evex_reg_VSHUFPS", &mycode, LIBXSMM_X86_INSTR_VSHUFPS, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSHUFPD", &mycode, LIBXSMM_X86_INSTR_VSHUFPD, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSHUFB", &mycode, LIBXSMM_X86_INSTR_VPSHUFB, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSHUFD", &mycode, LIBXSMM_X86_INSTR_VPSHUFD, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSHUFHW", &mycode, LIBXSMM_X86_INSTR_VPSHUFHW, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSHUFLW", &mycode, LIBXSMM_X86_INSTR_VPSHUFLW, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VUNPCKLPD", &mycode, LIBXSMM_X86_INSTR_VUNPCKLPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VUNPCKLPS", &mycode, LIBXSMM_X86_INSTR_VUNPCKLPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VUNPCKHPD", &mycode, LIBXSMM_X86_INSTR_VUNPCKHPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VUNPCKHPS", &mycode, LIBXSMM_X86_INSTR_VUNPCKHPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPUNPCKLBW", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLBW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPUNPCKHBW", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHBW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPUNPCKLWD", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLWD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPUNPCKHWD", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHWD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPUNPCKLDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLDQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPUNPCKHDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPUNPCKLQDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLQDQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPUNPCKHQDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMD", &mycode, LIBXSMM_X86_INSTR_VPERMD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMQ_I", &mycode, LIBXSMM_X86_INSTR_VPERMQ_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMPS", &mycode, LIBXSMM_X86_INSTR_VPERMPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMPD_I", &mycode, LIBXSMM_X86_INSTR_VPERMPD_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMILPS", &mycode, LIBXSMM_X86_INSTR_VPERMILPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMILPS_I", &mycode, LIBXSMM_X86_INSTR_VPERMILPS_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSHUFF32X4", &mycode, LIBXSMM_X86_INSTR_VSHUFF32X4, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSHUFF64X2", &mycode, LIBXSMM_X86_INSTR_VSHUFF64X2, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSHUFI32X4", &mycode, LIBXSMM_X86_INSTR_VSHUFI32X4, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSHUFI64X2", &mycode, LIBXSMM_X86_INSTR_VSHUFI64X2, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VEXTRACTF32X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF32X4, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VEXTRACTF64X2", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF64X2, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VEXTRACTF32X8", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF32X8, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VEXTRACTF64X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF64X4, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VEXTRACTI32X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI32X4, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VEXTRACTI64X2", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI64X2, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VEXTRACTI32X8", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI32X8, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VEXTRACTI64X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI64X4, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VINSERTI32X4", &mycode, LIBXSMM_X86_INSTR_VINSERTI32X4, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VBLENDMPS", &mycode, LIBXSMM_X86_INSTR_VBLENDMPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VBLENDMPD", &mycode, LIBXSMM_X86_INSTR_VBLENDMPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBLENDMB", &mycode, LIBXSMM_X86_INSTR_VPBLENDMB, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBLENDMW", &mycode, LIBXSMM_X86_INSTR_VPBLENDMW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBLENDMD", &mycode, LIBXSMM_X86_INSTR_VPBLENDMD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBLENDMQ", &mycode, LIBXSMM_X86_INSTR_VPBLENDMQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VEXPANDPD", &mycode, LIBXSMM_X86_INSTR_VEXPANDPD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VEXPANDPS", &mycode, LIBXSMM_X86_INSTR_VEXPANDPS, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPEXPANDQ", &mycode, LIBXSMM_X86_INSTR_VPEXPANDQ, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPEXPANDD", &mycode, LIBXSMM_X86_INSTR_VPEXPANDD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPEXPANDW", &mycode, LIBXSMM_X86_INSTR_VPEXPANDW, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPEXPANDB", &mycode, LIBXSMM_X86_INSTR_VPEXPANDB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMW", &mycode, LIBXSMM_X86_INSTR_VPERMW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMT2B", &mycode, LIBXSMM_X86_INSTR_VPERMT2B, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMT2W", &mycode, LIBXSMM_X86_INSTR_VPERMT2W, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMT2D", &mycode, LIBXSMM_X86_INSTR_VPERMT2D, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMT2Q", &mycode, LIBXSMM_X86_INSTR_VPERMT2Q, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMILPD", &mycode, LIBXSMM_X86_INSTR_VPERMILPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPERMILPD_I", &mycode, LIBXSMM_X86_INSTR_VPERMILPD_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD132PS", &mycode, LIBXSMM_X86_INSTR_VFMADD132PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD132PD", &mycode, LIBXSMM_X86_INSTR_VFMADD132PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD213PS", &mycode, LIBXSMM_X86_INSTR_VFMADD213PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD213PD", &mycode, LIBXSMM_X86_INSTR_VFMADD213PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD231PS", &mycode, LIBXSMM_X86_INSTR_VFMADD231PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD231PD", &mycode, LIBXSMM_X86_INSTR_VFMADD231PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB132PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB132PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB132PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB132PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB213PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB213PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB213PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB213PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB231PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB231PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB231PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB231PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD132PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD132PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD132PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD132PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD213PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD213PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD213PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD213PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD231PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD231PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD231PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD231PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB132PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB132PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB213PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB213PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB231PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB231PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231PD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD132SD", &mycode, LIBXSMM_X86_INSTR_VFMADD132SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD213SD", &mycode, LIBXSMM_X86_INSTR_VFMADD213SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD231SD", &mycode, LIBXSMM_X86_INSTR_VFMADD231SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD132SS", &mycode, LIBXSMM_X86_INSTR_VFMADD132SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD213SS", &mycode, LIBXSMM_X86_INSTR_VFMADD213SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD231SS", &mycode, LIBXSMM_X86_INSTR_VFMADD231SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB132SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB132SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB213SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB213SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB231SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB231SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB132SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB132SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB213SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB213SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB231SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB231SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD132SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD132SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD213SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD213SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD231SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD231SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD132SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD132SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD213SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD213SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD231SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD231SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB132SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB213SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB231SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB132SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB213SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB231SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRANGEPS", &mycode, LIBXSMM_X86_INSTR_VRANGEPS, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRANGEPD", &mycode, LIBXSMM_X86_INSTR_VRANGEPD, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRANGESS", &mycode, LIBXSMM_X86_INSTR_VRANGESS, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRANGESD", &mycode, LIBXSMM_X86_INSTR_VRANGESD, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VREDUCEPS", &mycode, LIBXSMM_X86_INSTR_VREDUCEPS, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VREDUCEPD", &mycode, LIBXSMM_X86_INSTR_VREDUCEPD, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VREDUCESS", &mycode, LIBXSMM_X86_INSTR_VREDUCESS, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VREDUCESD", &mycode, LIBXSMM_X86_INSTR_VREDUCESD, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRCP14PS", &mycode, LIBXSMM_X86_INSTR_VRCP14PS, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRCP14PD", &mycode, LIBXSMM_X86_INSTR_VRCP14PD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRCP14SS", &mycode, LIBXSMM_X86_INSTR_VRCP14SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRCP14SD", &mycode, LIBXSMM_X86_INSTR_VRCP14SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRNDSCALEPS", &mycode, LIBXSMM_X86_INSTR_VRNDSCALEPS, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRNDSCALEPD", &mycode, LIBXSMM_X86_INSTR_VRNDSCALEPD, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRNDSCALESS", &mycode, LIBXSMM_X86_INSTR_VRNDSCALESS, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRNDSCALESD", &mycode, LIBXSMM_X86_INSTR_VRNDSCALESD, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRSQRT14PS", &mycode, LIBXSMM_X86_INSTR_VRSQRT14PS, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRSQRT14PD", &mycode, LIBXSMM_X86_INSTR_VRSQRT14PD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRSQRT14SS", &mycode, LIBXSMM_X86_INSTR_VRSQRT14SS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRSQRT14SD", &mycode, LIBXSMM_X86_INSTR_VRSQRT14SD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSCALEFPS", &mycode, LIBXSMM_X86_INSTR_VSCALEFPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSCALEFPD", &mycode, LIBXSMM_X86_INSTR_VSCALEFPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSCALEFSS", &mycode, LIBXSMM_X86_INSTR_VSCALEFSS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSCALEFSD", &mycode, LIBXSMM_X86_INSTR_VSCALEFSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VCMPPS", &mycode, LIBXSMM_X86_INSTR_VCMPPS, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VCMPSS", &mycode, LIBXSMM_X86_INSTR_VCMPSS, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VCMPPD", &mycode, LIBXSMM_X86_INSTR_VCMPPD, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VCMPSD", &mycode, LIBXSMM_X86_INSTR_VCMPSD, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPB", &mycode, LIBXSMM_X86_INSTR_VPCMPB, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPUB", &mycode, LIBXSMM_X86_INSTR_VPCMPUB, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPW", &mycode, LIBXSMM_X86_INSTR_VPCMPW, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPUW", &mycode, LIBXSMM_X86_INSTR_VPCMPUW, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPD", &mycode, LIBXSMM_X86_INSTR_VPCMPD, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPUD", &mycode, LIBXSMM_X86_INSTR_VPCMPUD, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPQ", &mycode, LIBXSMM_X86_INSTR_VPCMPQ, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPUQ", &mycode, LIBXSMM_X86_INSTR_VPCMPUQ, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPEQB", &mycode, LIBXSMM_X86_INSTR_VPCMPEQB, 0, LIBXSMM_X86_IMM_UNDEF, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPEQW", &mycode, LIBXSMM_X86_INSTR_VPCMPEQD, 0, LIBXSMM_X86_IMM_UNDEF, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPEQD", &mycode, LIBXSMM_X86_INSTR_VPCMPEQW, 0, LIBXSMM_X86_IMM_UNDEF, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPEQQ", &mycode, LIBXSMM_X86_INSTR_VPCMPEQQ, 0, LIBXSMM_X86_IMM_UNDEF, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPGTB", &mycode, LIBXSMM_X86_INSTR_VPCMPGTB, 0, LIBXSMM_X86_IMM_UNDEF, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPGTW", &mycode, LIBXSMM_X86_INSTR_VPCMPGTD, 0, LIBXSMM_X86_IMM_UNDEF, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPGTD", &mycode, LIBXSMM_X86_INSTR_VPCMPGTW, 0, LIBXSMM_X86_IMM_UNDEF, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCMPGTQ", &mycode, LIBXSMM_X86_INSTR_VPCMPGTQ, 0, LIBXSMM_X86_IMM_UNDEF, 8 );
  test_evex_compute_3reg_general( "evex_reg_VCVTPS2PD", &mycode, LIBXSMM_X86_INSTR_VCVTPS2PD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VCVTPH2PS", &mycode, LIBXSMM_X86_INSTR_VCVTPH2PS, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VCVTPS2PH", &mycode, LIBXSMM_X86_INSTR_VCVTPS2PH, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VCVTDQ2PS", &mycode, LIBXSMM_X86_INSTR_VCVTDQ2PS, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VCVTPS2DQ", &mycode, LIBXSMM_X86_INSTR_VCVTPS2DQ, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VCVTPS2UDQ", &mycode, LIBXSMM_X86_INSTR_VCVTPS2UDQ, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVDW", &mycode, LIBXSMM_X86_INSTR_VPMOVDW, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVSXWD", &mycode, LIBXSMM_X86_INSTR_VPMOVSXWD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVDB", &mycode, LIBXSMM_X86_INSTR_VPMOVDB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVSDB", &mycode, LIBXSMM_X86_INSTR_VPMOVSDB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVUSDB", &mycode, LIBXSMM_X86_INSTR_VPMOVUSDB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVZXWD", &mycode, LIBXSMM_X86_INSTR_VPMOVZXWD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVZXBW", &mycode, LIBXSMM_X86_INSTR_VPMOVSXBW, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVSXBW", &mycode, LIBXSMM_X86_INSTR_VPMOVZXBW, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVSXBD", &mycode, LIBXSMM_X86_INSTR_VPMOVSXBD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVZXBD", &mycode, LIBXSMM_X86_INSTR_VPMOVZXBD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVUSWB", &mycode, LIBXSMM_X86_INSTR_VPMOVUSWB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVSWB", &mycode, LIBXSMM_X86_INSTR_VPMOVSWB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVWB", &mycode, LIBXSMM_X86_INSTR_VPMOVWB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSLLD_I", &mycode, LIBXSMM_X86_INSTR_VPSLLD_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSLLW_I", &mycode, LIBXSMM_X86_INSTR_VPSLLW_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSRAD_I", &mycode, LIBXSMM_X86_INSTR_VPSRAD_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSRAW_I", &mycode, LIBXSMM_X86_INSTR_VPSRAW_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSRLD_I", &mycode, LIBXSMM_X86_INSTR_VPSRLD_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSLLVW", &mycode, LIBXSMM_X86_INSTR_VPSLLVW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSLLVD", &mycode, LIBXSMM_X86_INSTR_VPSLLVD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSLLVQ", &mycode, LIBXSMM_X86_INSTR_VPSLLVQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSRAVW", &mycode, LIBXSMM_X86_INSTR_VPSRAVW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSRAVD", &mycode, LIBXSMM_X86_INSTR_VPSRAVD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSRAVQ", &mycode, LIBXSMM_X86_INSTR_VPSRAVQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSRLVW", &mycode, LIBXSMM_X86_INSTR_VPSRLVW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSRLVD", &mycode, LIBXSMM_X86_INSTR_VPSRLVD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSRLVQ", &mycode, LIBXSMM_X86_INSTR_VPSRLVQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VXORPD", &mycode, LIBXSMM_X86_INSTR_VXORPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VADDPD", &mycode, LIBXSMM_X86_INSTR_VADDPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMULPD", &mycode, LIBXSMM_X86_INSTR_VMULPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSUBPD", &mycode, LIBXSMM_X86_INSTR_VSUBPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VDIVPD", &mycode, LIBXSMM_X86_INSTR_VDIVPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMINPD", &mycode, LIBXSMM_X86_INSTR_VMINPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMAXPD", &mycode, LIBXSMM_X86_INSTR_VMAXPD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSQRTPD", &mycode, LIBXSMM_X86_INSTR_VSQRTPD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VADDSD", &mycode, LIBXSMM_X86_INSTR_VADDSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMULSD", &mycode, LIBXSMM_X86_INSTR_VMULSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSUBSD", &mycode, LIBXSMM_X86_INSTR_VSUBSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VDIVSD", &mycode, LIBXSMM_X86_INSTR_VDIVSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMINSD", &mycode, LIBXSMM_X86_INSTR_VMINSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMAXSD", &mycode, LIBXSMM_X86_INSTR_VMAXSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSQRTSD", &mycode, LIBXSMM_X86_INSTR_VSQRTSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VXORPS", &mycode, LIBXSMM_X86_INSTR_VXORPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VADDPS", &mycode, LIBXSMM_X86_INSTR_VADDPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMULPS", &mycode, LIBXSMM_X86_INSTR_VMULPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSUBPS", &mycode, LIBXSMM_X86_INSTR_VSUBPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VDIVPS", &mycode, LIBXSMM_X86_INSTR_VDIVPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMINPS", &mycode, LIBXSMM_X86_INSTR_VMINPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMAXPS", &mycode, LIBXSMM_X86_INSTR_VMAXPS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSQRTPS", &mycode, LIBXSMM_X86_INSTR_VSQRTPS, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMULSS", &mycode, LIBXSMM_X86_INSTR_VMULSS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VADDSS", &mycode, LIBXSMM_X86_INSTR_VADDSS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSUBSS", &mycode, LIBXSMM_X86_INSTR_VSUBSS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VDIVSS", &mycode, LIBXSMM_X86_INSTR_VDIVSS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMINSS", &mycode, LIBXSMM_X86_INSTR_VMINSS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMAXSS", &mycode, LIBXSMM_X86_INSTR_VMAXSS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSQRTSS", &mycode, LIBXSMM_X86_INSTR_VSQRTSS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPXORD", &mycode, LIBXSMM_X86_INSTR_VPXORD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPORD", &mycode, LIBXSMM_X86_INSTR_VPORD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPXORQ", &mycode, LIBXSMM_X86_INSTR_VPXORQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPORQ", &mycode, LIBXSMM_X86_INSTR_VPORQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPANDD", &mycode, LIBXSMM_X86_INSTR_VPANDD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPANDQ", &mycode, LIBXSMM_X86_INSTR_VPANDQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPADDQ", &mycode, LIBXSMM_X86_INSTR_VPADDQ, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPADDB", &mycode, LIBXSMM_X86_INSTR_VPADDB, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPADDW", &mycode, LIBXSMM_X86_INSTR_VPADDW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPADDD", &mycode, LIBXSMM_X86_INSTR_VPADDD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMADDWD", &mycode, LIBXSMM_X86_INSTR_VPMADDWD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMADDUBSW", &mycode, LIBXSMM_X86_INSTR_VPMADDUBSW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPADDSW", &mycode, LIBXSMM_X86_INSTR_VPADDSW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPADDSB", &mycode, LIBXSMM_X86_INSTR_VPADDSB, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSUBD", &mycode, LIBXSMM_X86_INSTR_VPSUBD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMAXSD", &mycode, LIBXSMM_X86_INSTR_VPMAXSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMAXSW", &mycode, LIBXSMM_X86_INSTR_VPMAXSW, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMINSD", &mycode, LIBXSMM_X86_INSTR_VPMINSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPDPBUSD", &mycode, LIBXSMM_X86_INSTR_VPDPBUSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPDPBUSDS", &mycode, LIBXSMM_X86_INSTR_VPDPBUSDS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPDPWSSD", &mycode, LIBXSMM_X86_INSTR_VPDPWSSD, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPDPWSSDS", &mycode, LIBXSMM_X86_INSTR_VPDPWSSDS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VDPBF16PS", &mycode, LIBXSMM_X86_INSTR_VDPBF16PS, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VCVTNEPS2BF16", &mycode, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VCVTNE2PS2BF16", &mycode, LIBXSMM_X86_INSTR_VCVTNE2PS2BF16, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMOVDQU64_LD", &mycode, LIBXSMM_X86_INSTR_VMOVDQU64_LD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMOVDQU64_ST", &mycode, LIBXSMM_X86_INSTR_VMOVDQU64_ST, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
#if 0
  test_evex_compute_3reg_general( "evex_reg_VMOVD_LD", &mycode, LIBXSMM_X86_INSTR_VMOVD_LD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMOVD_ST", &mycode, LIBXSMM_X86_INSTR_VMOVD_ST, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMOVQ_LD", &mycode, LIBXSMM_X86_INSTR_VMOVD_LD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMOVQ_ST", &mycode, LIBXSMM_X86_INSTR_VMOVD_ST, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
#endif
  test_evex_compute_3reg_general( "evex_reg_VPBROADCASTB_GPR", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTB_GPR, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBROADCASTW_GPR", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTW_GPR, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBROADCASTD_GPR", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTD_GPR, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBROADCASTQ_GPR", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTQ_GPR, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VBROADCASTSS", &mycode, LIBXSMM_X86_INSTR_VBROADCASTSS, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VBROADCASTSD", &mycode, LIBXSMM_X86_INSTR_VBROADCASTSD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBROADCASTB", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBROADCASTW", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTW, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBROADCASTD", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPBROADCASTQ", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTQ, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VBROADCASTI32X2", &mycode, LIBXSMM_X86_INSTR_VBROADCASTI32X2, 1, LIBXSMM_X86_IMM_UNDEF, 32 );

  /* AVX512 FP16 */
  test_evex_compute_3reg_general( "evex_reg_VADDPH", &mycode, LIBXSMM_X86_INSTR_VADDPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VADDSH", &mycode, LIBXSMM_X86_INSTR_VADDSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VCMPPH", &mycode, LIBXSMM_X86_INSTR_VCMPPH, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VCMPSH", &mycode, LIBXSMM_X86_INSTR_VCMPSH, 0, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VDIVPH", &mycode, LIBXSMM_X86_INSTR_VDIVPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VDIVSH", &mycode, LIBXSMM_X86_INSTR_VDIVSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFCMADDCPH", &mycode, LIBXSMM_X86_INSTR_VFCMADDCPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADDCPH", &mycode, LIBXSMM_X86_INSTR_VFMADDCPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFCMADDCSH", &mycode, LIBXSMM_X86_INSTR_VFCMADDCSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADDCSH", &mycode, LIBXSMM_X86_INSTR_VFMADDCSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFCMULCPH", &mycode, LIBXSMM_X86_INSTR_VFCMULCPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMULCPH", &mycode, LIBXSMM_X86_INSTR_VFMULCPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFCMULCSH", &mycode, LIBXSMM_X86_INSTR_VFCMULCSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMULCSH", &mycode, LIBXSMM_X86_INSTR_VFMULCSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADDSUB132PH", &mycode, LIBXSMM_X86_INSTR_VFMADDSUB132PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADDSUB213PH", &mycode, LIBXSMM_X86_INSTR_VFMADDSUB213PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADDSUB231PH", &mycode, LIBXSMM_X86_INSTR_VFMADDSUB231PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUBADD132PH", &mycode, LIBXSMM_X86_INSTR_VFMSUBADD132PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUBADD213PH", &mycode, LIBXSMM_X86_INSTR_VFMSUBADD213PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUBADD231PH", &mycode, LIBXSMM_X86_INSTR_VFMSUBADD231PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD132PH", &mycode, LIBXSMM_X86_INSTR_VFMADD132PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD213PH", &mycode, LIBXSMM_X86_INSTR_VFMADD213PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD231PH", &mycode, LIBXSMM_X86_INSTR_VFMADD231PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD132PH", &mycode, LIBXSMM_X86_INSTR_VFNMADD132PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD213PH", &mycode, LIBXSMM_X86_INSTR_VFNMADD213PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD231PH", &mycode, LIBXSMM_X86_INSTR_VFNMADD231PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD132SH", &mycode, LIBXSMM_X86_INSTR_VFMADD132SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD213SH", &mycode, LIBXSMM_X86_INSTR_VFMADD213SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMADD231SH", &mycode, LIBXSMM_X86_INSTR_VFMADD231SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD132SH", &mycode, LIBXSMM_X86_INSTR_VFNMADD132SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD213SH", &mycode, LIBXSMM_X86_INSTR_VFNMADD213SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMADD231SH", &mycode, LIBXSMM_X86_INSTR_VFNMADD231SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB132PH", &mycode, LIBXSMM_X86_INSTR_VFMSUB132PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB213PH", &mycode, LIBXSMM_X86_INSTR_VFMSUB213PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB231PH", &mycode, LIBXSMM_X86_INSTR_VFMSUB231PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB132PH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB213PH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB231PH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231PH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB132SH", &mycode, LIBXSMM_X86_INSTR_VFMSUB132SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB213SH", &mycode, LIBXSMM_X86_INSTR_VFMSUB213SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFMSUB231SH", &mycode, LIBXSMM_X86_INSTR_VFMSUB231SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB132SH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB213SH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VFNMSUB231SH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231SH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPCLASSPH", &mycode, LIBXSMM_X86_INSTR_VPCLASSPH, 1, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VPCLASSSH", &mycode, LIBXSMM_X86_INSTR_VPCLASSSH, 1, 0x01, 8 );
  test_evex_compute_3reg_general( "evex_reg_VGETEXPPH", &mycode, LIBXSMM_X86_INSTR_VGETEXPPH, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VGETEXPSH", &mycode, LIBXSMM_X86_INSTR_VGETEXPSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VGETMANTPH", &mycode, LIBXSMM_X86_INSTR_VGETMANTPH, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VGETMANTSH", &mycode, LIBXSMM_X86_INSTR_VGETMANTSH, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMAXPH", &mycode, LIBXSMM_X86_INSTR_VMAXPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMAXSH", &mycode, LIBXSMM_X86_INSTR_VMAXSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMINPH", &mycode, LIBXSMM_X86_INSTR_VMINPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMINSH", &mycode, LIBXSMM_X86_INSTR_VMINSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMOVSH_LD_3REG", &mycode, LIBXSMM_X86_INSTR_VMOVSH_LD_3REG, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMOVSH_ST_3REG", &mycode, LIBXSMM_X86_INSTR_VMOVSH_ST_3REG, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
#if 0
  test_evex_compute_3reg_general( "evex_reg_VMOVW_LD", &mycode, LIBXSMM_X86_INSTR_VMOVW_LD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMOVW_ST", &mycode, LIBXSMM_X86_INSTR_VMOVW_ST, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
#endif
  test_evex_compute_3reg_general( "evex_reg_VMULPH", &mycode, LIBXSMM_X86_INSTR_VMULPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VMULSH", &mycode, LIBXSMM_X86_INSTR_VMULSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRCPPH", &mycode, LIBXSMM_X86_INSTR_VRCPPH, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRCPSH", &mycode, LIBXSMM_X86_INSTR_VRCPSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VREDUCEPH", &mycode, LIBXSMM_X86_INSTR_VREDUCEPH, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VREDUCESH", &mycode, LIBXSMM_X86_INSTR_VREDUCESH, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRNDSCALEPH", &mycode, LIBXSMM_X86_INSTR_VRNDSCALEPH, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRNDSCALESH", &mycode, LIBXSMM_X86_INSTR_VRNDSCALESH, 0, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRSQRTPH", &mycode, LIBXSMM_X86_INSTR_VRSQRTPH, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VRSQRTSH", &mycode, LIBXSMM_X86_INSTR_VRSQRTSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSCALEFPH", &mycode, LIBXSMM_X86_INSTR_VSCALEFPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSCALEFSH", &mycode, LIBXSMM_X86_INSTR_VSCALEFSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSQRTPH", &mycode, LIBXSMM_X86_INSTR_VSQRTPH, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSQRTSH", &mycode, LIBXSMM_X86_INSTR_VSQRTSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSUBPH", &mycode, LIBXSMM_X86_INSTR_VSUBPH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VSUBSH", &mycode, LIBXSMM_X86_INSTR_VSUBSH, 0, LIBXSMM_X86_IMM_UNDEF, 32 );

  /* VEX only encodings, even on EVEX machine */
  test_vex_compute_3reg_general( "vex_reg_VPERM2F128", &mycode, LIBXSMM_X86_INSTR_VPERM2F128, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPERM2I128", &mycode, LIBXSMM_X86_INSTR_VPERM2I128, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_EXTRACTF128", &mycode, LIBXSMM_X86_INSTR_VPERM2F128, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_EXTRACTI128", &mycode, LIBXSMM_X86_INSTR_VPERM2I128, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPERMILPD_VEX", &mycode, LIBXSMM_X86_INSTR_VPERMILPD_VEX, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPERMILPD_VEX_I", &mycode, LIBXSMM_X86_INSTR_VPERMILPD_VEX_I, 1, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VBLENDPD", &mycode, LIBXSMM_X86_INSTR_VBLENDPD, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VBLENDPS", &mycode, LIBXSMM_X86_INSTR_VBLENDPS, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VBLENDVPD", &mycode, LIBXSMM_X86_INSTR_VBLENDVPD, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VBLENDVPS", &mycode, LIBXSMM_X86_INSTR_VBLENDVPS, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPBLENDD", &mycode, LIBXSMM_X86_INSTR_VPBLENDD, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPBLENDW", &mycode, LIBXSMM_X86_INSTR_VPBLENDW, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPBLENDVB", &mycode, LIBXSMM_X86_INSTR_VPBLENDVB, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VMOVMSKPD", &mycode, LIBXSMM_X86_INSTR_VMOVMSKPD, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VMOVMSKPS", &mycode, LIBXSMM_X86_INSTR_VMOVMSKPS, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPMOVMSKB", &mycode, LIBXSMM_X86_INSTR_VPMOVMSKB, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VROUNDPD", &mycode, LIBXSMM_X86_INSTR_VROUNDPD, 1, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VROUNDSD", &mycode, LIBXSMM_X86_INSTR_VROUNDSD, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VROUNDPS", &mycode, LIBXSMM_X86_INSTR_VROUNDPS, 1, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VROUNDSS", &mycode, LIBXSMM_X86_INSTR_VROUNDSS, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VRCPPS", &mycode, LIBXSMM_X86_INSTR_VRCPPS, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VRCPSS", &mycode, LIBXSMM_X86_INSTR_VRCPSS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VRSQRTPS", &mycode, LIBXSMM_X86_INSTR_VRSQRTPS, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VRSQRTSS", &mycode, LIBXSMM_X86_INSTR_VRSQRTSS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPESTRI", &mycode, LIBXSMM_X86_INSTR_VPCMPESTRI, 1, 0x01, 1 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPESTRM", &mycode, LIBXSMM_X86_INSTR_VPCMPESTRM, 1, 0x01, 1 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPISTRI", &mycode, LIBXSMM_X86_INSTR_VPCMPISTRI, 1, 0x01, 1 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPISTRM", &mycode, LIBXSMM_X86_INSTR_VPCMPISTRM, 1, 0x01, 1 );
  test_vex_compute_3reg_general( "vex_reg_VBROADCASTSD_VEX", &mycode, LIBXSMM_X86_INSTR_VBROADCASTSD_VEX, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPBROADCASTQ_VEX", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTQ_VEX, 1, LIBXSMM_X86_IMM_UNDEF, 0 );

  /* VEX only */
  mycode.arch = LIBXSMM_X86_AVX2_SRF;
  test_vex_compute_3reg_general( "vex_reg_VMOVD_LD", &mycode, LIBXSMM_X86_INSTR_VMOVD_LD, 1, LIBXSMM_X86_IMM_UNDEF, 1 );
  test_vex_compute_3reg_general( "vex_reg_VMOVD_ST", &mycode, LIBXSMM_X86_INSTR_VMOVD_ST, 1, LIBXSMM_X86_IMM_UNDEF, 1 );
  test_vex_compute_3reg_general( "vex_reg_VMOVQ_LD", &mycode, LIBXSMM_X86_INSTR_VMOVQ_LD, 1, LIBXSMM_X86_IMM_UNDEF, 1 );
  test_vex_compute_3reg_general( "vex_reg_VMOVQ_ST", &mycode, LIBXSMM_X86_INSTR_VMOVQ_ST, 1, LIBXSMM_X86_IMM_UNDEF, 1 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPEQB", &mycode, LIBXSMM_X86_INSTR_VPCMPEQB, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPEQW", &mycode, LIBXSMM_X86_INSTR_VPCMPEQW, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPEQD", &mycode, LIBXSMM_X86_INSTR_VPCMPEQD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPEQQ", &mycode, LIBXSMM_X86_INSTR_VPCMPEQQ, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPGTB", &mycode, LIBXSMM_X86_INSTR_VPCMPGTB, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPGTW", &mycode, LIBXSMM_X86_INSTR_VPCMPGTW, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPGTD", &mycode, LIBXSMM_X86_INSTR_VPCMPGTD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPGTQ", &mycode, LIBXSMM_X86_INSTR_VPCMPGTQ, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VBROADCASTSS", &mycode, LIBXSMM_X86_INSTR_VBROADCASTSS, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPBROADCASTB", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTB, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPBROADCASTW", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTW, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPBROADCASTD", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTD, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD132PS", &mycode, LIBXSMM_X86_INSTR_VFMADD132PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD132PD", &mycode, LIBXSMM_X86_INSTR_VFMADD132PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD213PS", &mycode, LIBXSMM_X86_INSTR_VFMADD213PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD213PD", &mycode, LIBXSMM_X86_INSTR_VFMADD213PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD231PS", &mycode, LIBXSMM_X86_INSTR_VFMADD231PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD231PD", &mycode, LIBXSMM_X86_INSTR_VFMADD231PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB132PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB132PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB132PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB132PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB213PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB213PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB213PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB213PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB231PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB231PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB231PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB231PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD132PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD132PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD132PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD132PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD213PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD213PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD213PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD213PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD231PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD231PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD231PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD231PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB132PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB132PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB213PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB213PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB231PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231PS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB231PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231PD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD132SD", &mycode, LIBXSMM_X86_INSTR_VFMADD132SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD213SD", &mycode, LIBXSMM_X86_INSTR_VFMADD213SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD231SD", &mycode, LIBXSMM_X86_INSTR_VFMADD231SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD132SS", &mycode, LIBXSMM_X86_INSTR_VFMADD132SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD213SS", &mycode, LIBXSMM_X86_INSTR_VFMADD213SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMADD231SS", &mycode, LIBXSMM_X86_INSTR_VFMADD231SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB132SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB132SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB213SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB213SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB231SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB231SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB132SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB132SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB213SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB213SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFMSUB231SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB231SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD132SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD132SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD213SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD213SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD231SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD231SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD132SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD132SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD213SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD213SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMADD231SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD231SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB132SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB213SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB231SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231SD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB132SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB213SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VFNMSUB231SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231SS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPDPWSSD", &mycode, LIBXSMM_X86_INSTR_VPDPWSSD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPDPBUSD", &mycode, LIBXSMM_X86_INSTR_VPDPBUSD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPDPWSSDS", &mycode, LIBXSMM_X86_INSTR_VPDPWSSDS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPDPBUSDS", &mycode, LIBXSMM_X86_INSTR_VPDPBUSDS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPADDD", &mycode, LIBXSMM_X86_INSTR_VPADDD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPMADDUBSW", &mycode, LIBXSMM_X86_INSTR_VPMADDUBSW, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPMADDWD", &mycode, LIBXSMM_X86_INSTR_VPMADDWD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPDPBSUD", &mycode, LIBXSMM_X86_INSTR_VPDPBSUD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPDPBSUDS", &mycode, LIBXSMM_X86_INSTR_VPDPBSUDS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPDPBSSD", &mycode, LIBXSMM_X86_INSTR_VPDPBSSD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPDPBSSDS", &mycode, LIBXSMM_X86_INSTR_VPDPBSSDS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPDPBUUD", &mycode, LIBXSMM_X86_INSTR_VPDPBUUD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VPDPBUUDS", &mycode, LIBXSMM_X86_INSTR_VPDPBUUDS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VCVTNEPS2BF16", &mycode, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, 1, LIBXSMM_X86_IMM_UNDEF, 0 );

  test_vex_compute_mem_2reg_general( "vex_mem_VPDPBUSD", &mycode, LIBXSMM_X86_INSTR_VPDPBUSD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_mem_2reg_general( "vex_mem_VPDPBUSDS", &mycode, LIBXSMM_X86_INSTR_VPDPBUSDS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_mem_2reg_general( "vex_mem_VPDPBSUD", &mycode, LIBXSMM_X86_INSTR_VPDPBSUD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_mem_2reg_general( "vex_mem_VPDPBSUDS", &mycode, LIBXSMM_X86_INSTR_VPDPBSUDS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_mem_2reg_general( "vex_mem_VPDPBSSD", &mycode, LIBXSMM_X86_INSTR_VPDPBSSD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_mem_2reg_general( "vex_mem_VPDPBSSDS", &mycode, LIBXSMM_X86_INSTR_VPDPBSSDS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_mem_2reg_general( "vex_mem_VPDPBUUD", &mycode, LIBXSMM_X86_INSTR_VPDPBUUD, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_mem_2reg_general( "vex_mem_VPDPBUUDS", &mycode, LIBXSMM_X86_INSTR_VPDPBUUDS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_mem_2reg_general( "vex_mem_VCVTNEPS2BF16", &mycode, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, 1, LIBXSMM_X86_IMM_UNDEF, 0 );

  /* testing compute mem-reg instructions */
  mycode.arch = LIBXSMM_X86_AVX512_GNR;
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFPS", &mycode, LIBXSMM_X86_INSTR_VSHUFPS, 2, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFPD", &mycode, LIBXSMM_X86_INSTR_VSHUFPD, 2, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSHUFB", &mycode, LIBXSMM_X86_INSTR_VPSHUFB, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSHUFD", &mycode, LIBXSMM_X86_INSTR_VPSHUFD, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSHUFHW", &mycode, LIBXSMM_X86_INSTR_VPSHUFHW, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSHUFLW", &mycode, LIBXSMM_X86_INSTR_VPSHUFLW, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFF32X4", &mycode, LIBXSMM_X86_INSTR_VSHUFF32X4, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFF64X2", &mycode, LIBXSMM_X86_INSTR_VSHUFF64X2, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFI32X4", &mycode, LIBXSMM_X86_INSTR_VSHUFI32X4, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFI64X2", &mycode, LIBXSMM_X86_INSTR_VSHUFI64X2, 2, 0x01, 32, 0, 1 );
  test_vex_compute_mem_2reg_general( "evex_mem_VEXTRACTF128", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF128, 1, 0x01, 0 );
  test_vex_compute_mem_2reg_general( "evex_mem_VEXTRACTI128", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI128, 1, 0x01, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTF32X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF32X4, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTF64X2", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF64X2, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTF32X8", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF32X8, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTF64X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF64X4, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTI32X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI32X4, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTI64X2", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI64X2, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTI32X8", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI32X8, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTI64X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI64X4, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VINSERTI32X4", &mycode, LIBXSMM_X86_INSTR_VINSERTI32X4, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VBLENDMPS", &mycode, LIBXSMM_X86_INSTR_VBLENDMPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VBLENDMPD", &mycode, LIBXSMM_X86_INSTR_VBLENDMPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPBLENDMB", &mycode, LIBXSMM_X86_INSTR_VPBLENDMB, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPBLENDMW", &mycode, LIBXSMM_X86_INSTR_VPBLENDMW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPBLENDMD", &mycode, LIBXSMM_X86_INSTR_VPBLENDMD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPBLENDMQ", &mycode, LIBXSMM_X86_INSTR_VPBLENDMQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXPANDPD", &mycode, LIBXSMM_X86_INSTR_VEXPANDPD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXPANDPS", &mycode, LIBXSMM_X86_INSTR_VEXPANDPS, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPEXPANDQ", &mycode, LIBXSMM_X86_INSTR_VPEXPANDQ, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPEXPANDD", &mycode, LIBXSMM_X86_INSTR_VPEXPANDD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPEXPANDW", &mycode, LIBXSMM_X86_INSTR_VPEXPANDW, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPEXPANDB", &mycode, LIBXSMM_X86_INSTR_VPEXPANDB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VUNPCKLPD", &mycode, LIBXSMM_X86_INSTR_VUNPCKLPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VUNPCKLPS", &mycode, LIBXSMM_X86_INSTR_VUNPCKLPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VUNPCKHPD", &mycode, LIBXSMM_X86_INSTR_VUNPCKHPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VUNPCKHPS", &mycode, LIBXSMM_X86_INSTR_VUNPCKHPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKLBW", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLBW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKHBW", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHBW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKLWD", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLWD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKHWD", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHWD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKLDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKHDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKLQDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLQDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKHQDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_vex_compute_mem_2reg_general( "evex_mem_VPERM2F128", &mycode, LIBXSMM_X86_INSTR_VPERM2F128, 2, 0x01, 0 );
  test_vex_compute_mem_2reg_general( "evex_mem_VPERM2I128", &mycode, LIBXSMM_X86_INSTR_VPERM2I128, 2, 0x01, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMW", &mycode, LIBXSMM_X86_INSTR_VPERMW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMD", &mycode, LIBXSMM_X86_INSTR_VPERMD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMQ_I", &mycode, LIBXSMM_X86_INSTR_VPERMQ_I, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMT2B", &mycode, LIBXSMM_X86_INSTR_VPERMT2B, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMT2W", &mycode, LIBXSMM_X86_INSTR_VPERMT2W, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMT2D", &mycode, LIBXSMM_X86_INSTR_VPERMT2D, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMT2Q", &mycode, LIBXSMM_X86_INSTR_VPERMT2Q, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD132PS", &mycode, LIBXSMM_X86_INSTR_VFMADD132PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD132PD", &mycode, LIBXSMM_X86_INSTR_VFMADD132PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD213PS", &mycode, LIBXSMM_X86_INSTR_VFMADD213PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD213PD", &mycode, LIBXSMM_X86_INSTR_VFMADD213PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD231PS", &mycode, LIBXSMM_X86_INSTR_VFMADD231PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD231PD", &mycode, LIBXSMM_X86_INSTR_VFMADD231PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB132PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB132PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB132PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB132PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB213PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB213PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB213PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB213PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB231PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB231PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB231PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB231PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD132PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD132PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD132PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD132PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD213PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD213PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD213PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD213PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD231PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD231PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD231PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD231PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB132PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB132PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB213PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB213PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB231PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB231PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD132SD", &mycode, LIBXSMM_X86_INSTR_VFMADD132SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD213SD", &mycode, LIBXSMM_X86_INSTR_VFMADD213SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD231SD", &mycode, LIBXSMM_X86_INSTR_VFMADD231SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD132SS", &mycode, LIBXSMM_X86_INSTR_VFMADD132SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD213SS", &mycode, LIBXSMM_X86_INSTR_VFMADD213SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD231SS", &mycode, LIBXSMM_X86_INSTR_VFMADD231SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB132SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB132SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB213SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB213SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB231SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB231SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB132SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB132SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB213SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB213SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB231SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB231SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD132SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD132SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD213SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD213SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD231SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD231SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD132SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD132SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD213SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD213SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD231SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD231SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB132SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB213SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB231SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB132SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB213SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB231SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRANGEPS", &mycode, LIBXSMM_X86_INSTR_VRANGEPS, 2, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRANGEPD", &mycode, LIBXSMM_X86_INSTR_VRANGEPD, 2, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRANGESS", &mycode, LIBXSMM_X86_INSTR_VRANGESS, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRANGESD", &mycode, LIBXSMM_X86_INSTR_VRANGESD, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VREDUCEPS", &mycode, LIBXSMM_X86_INSTR_VREDUCEPS, 1, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VREDUCEPD", &mycode, LIBXSMM_X86_INSTR_VREDUCEPD, 1, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VREDUCESS", &mycode, LIBXSMM_X86_INSTR_VREDUCESS, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VREDUCESD", &mycode, LIBXSMM_X86_INSTR_VREDUCESD, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRCP14PS", &mycode, LIBXSMM_X86_INSTR_VRCP14PS, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRCP14PD", &mycode, LIBXSMM_X86_INSTR_VRCP14PD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRCP14SS", &mycode, LIBXSMM_X86_INSTR_VRCP14SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRCP14SD", &mycode, LIBXSMM_X86_INSTR_VRCP14SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRNDSCALEPS", &mycode, LIBXSMM_X86_INSTR_VRNDSCALEPS, 1, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRNDSCALEPD", &mycode, LIBXSMM_X86_INSTR_VRNDSCALEPD, 1, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRNDSCALESS", &mycode, LIBXSMM_X86_INSTR_VRNDSCALESS, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRNDSCALESD", &mycode, LIBXSMM_X86_INSTR_VRNDSCALESD, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRSQRT14PS", &mycode, LIBXSMM_X86_INSTR_VRSQRT14PS, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRSQRT14PD", &mycode, LIBXSMM_X86_INSTR_VRSQRT14PD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRSQRT14SS", &mycode, LIBXSMM_X86_INSTR_VRSQRT14SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRSQRT14SD", &mycode, LIBXSMM_X86_INSTR_VRSQRT14SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSCALEFPS", &mycode, LIBXSMM_X86_INSTR_VSCALEFPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSCALEFPD", &mycode, LIBXSMM_X86_INSTR_VSCALEFPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSCALEFSS", &mycode, LIBXSMM_X86_INSTR_VSCALEFSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSCALEFSD", &mycode, LIBXSMM_X86_INSTR_VSCALEFSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCMPPS", &mycode, LIBXSMM_X86_INSTR_VCMPPS, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCMPSS", &mycode, LIBXSMM_X86_INSTR_VCMPSS, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCMPPD", &mycode, LIBXSMM_X86_INSTR_VCMPPD, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCMPSD", &mycode, LIBXSMM_X86_INSTR_VCMPSD, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPB", &mycode, LIBXSMM_X86_INSTR_VPCMPB, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPUB", &mycode, LIBXSMM_X86_INSTR_VPCMPUB, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPW", &mycode, LIBXSMM_X86_INSTR_VPCMPW, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPUW", &mycode, LIBXSMM_X86_INSTR_VPCMPUW, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPD", &mycode, LIBXSMM_X86_INSTR_VPCMPD, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPUD", &mycode, LIBXSMM_X86_INSTR_VPCMPUD, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPQ", &mycode, LIBXSMM_X86_INSTR_VPCMPQ, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPUQ", &mycode, LIBXSMM_X86_INSTR_VPCMPUQ, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPEQB", &mycode, LIBXSMM_X86_INSTR_VPCMPEQB, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPEQW", &mycode, LIBXSMM_X86_INSTR_VPCMPEQW, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPEQD", &mycode, LIBXSMM_X86_INSTR_VPCMPEQD, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPEQQ", &mycode, LIBXSMM_X86_INSTR_VPCMPEQQ, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPGTB", &mycode, LIBXSMM_X86_INSTR_VPCMPGTB, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPGTW", &mycode, LIBXSMM_X86_INSTR_VPCMPGTW, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPGTD", &mycode, LIBXSMM_X86_INSTR_VPCMPGTD, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPGTQ", &mycode, LIBXSMM_X86_INSTR_VPCMPGTQ, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTPS2PD", &mycode, LIBXSMM_X86_INSTR_VCVTPS2PD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTPH2PS", &mycode, LIBXSMM_X86_INSTR_VCVTPH2PS, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTPS2PH", &mycode, LIBXSMM_X86_INSTR_VCVTPS2PH, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTDQ2PS", &mycode, LIBXSMM_X86_INSTR_VCVTDQ2PS, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTPS2DQ", &mycode, LIBXSMM_X86_INSTR_VCVTPS2DQ, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTPS2UDQ", &mycode, LIBXSMM_X86_INSTR_VCVTPS2UDQ, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVDW", &mycode, LIBXSMM_X86_INSTR_VPMOVDW, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVSXWD", &mycode, LIBXSMM_X86_INSTR_VPMOVSXWD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVDB", &mycode, LIBXSMM_X86_INSTR_VPMOVDB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVSDB", &mycode, LIBXSMM_X86_INSTR_VPMOVSDB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVUSDB", &mycode, LIBXSMM_X86_INSTR_VPMOVUSDB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVZXWD", &mycode, LIBXSMM_X86_INSTR_VPMOVZXWD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVZXBW", &mycode, LIBXSMM_X86_INSTR_VPMOVZXBW, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVSXBW", &mycode, LIBXSMM_X86_INSTR_VPMOVSXBW, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVSXBD", &mycode, LIBXSMM_X86_INSTR_VPMOVSXBD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVZXBD", &mycode, LIBXSMM_X86_INSTR_VPMOVZXBD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVUSWB", &mycode, LIBXSMM_X86_INSTR_VPMOVUSWB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVSWB", &mycode, LIBXSMM_X86_INSTR_VPMOVSWB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVWB", &mycode, LIBXSMM_X86_INSTR_VPMOVWB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSLLD_I", &mycode, LIBXSMM_X86_INSTR_VPSLLD_I, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSLLW_I", &mycode, LIBXSMM_X86_INSTR_VPSLLW_I, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSRAD_I", &mycode, LIBXSMM_X86_INSTR_VPSRAD_I, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSRAW_I", &mycode, LIBXSMM_X86_INSTR_VPSRAW_I, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSRLD_I", &mycode, LIBXSMM_X86_INSTR_VPSRLD_I, 1, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSRAVD", &mycode, LIBXSMM_X86_INSTR_VPSRAVD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VXORPD", &mycode, LIBXSMM_X86_INSTR_VXORPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VADDPD", &mycode, LIBXSMM_X86_INSTR_VADDPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMULPD", &mycode, LIBXSMM_X86_INSTR_VMULPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSUBPD", &mycode, LIBXSMM_X86_INSTR_VSUBPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VDIVPD", &mycode, LIBXSMM_X86_INSTR_VDIVPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMAXPD", &mycode, LIBXSMM_X86_INSTR_VMAXPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VADDSD", &mycode, LIBXSMM_X86_INSTR_VADDSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMULSD", &mycode, LIBXSMM_X86_INSTR_VMULSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSUBSD", &mycode, LIBXSMM_X86_INSTR_VSUBSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VXORPS", &mycode, LIBXSMM_X86_INSTR_VXORPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VADDPS", &mycode, LIBXSMM_X86_INSTR_VADDPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMULPS", &mycode, LIBXSMM_X86_INSTR_VMULPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSUBPS", &mycode, LIBXSMM_X86_INSTR_VSUBPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VDIVPS", &mycode, LIBXSMM_X86_INSTR_VDIVPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMAXPS", &mycode, LIBXSMM_X86_INSTR_VMAXPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMULSS", &mycode, LIBXSMM_X86_INSTR_VMULSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VADDSS", &mycode, LIBXSMM_X86_INSTR_VADDSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSUBSS", &mycode, LIBXSMM_X86_INSTR_VSUBSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPXORD", &mycode, LIBXSMM_X86_INSTR_VPXORD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPORD", &mycode, LIBXSMM_X86_INSTR_VPORD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPXORQ", &mycode, LIBXSMM_X86_INSTR_VPXORQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPORQ", &mycode, LIBXSMM_X86_INSTR_VPORQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPANDD", &mycode, LIBXSMM_X86_INSTR_VPANDD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPANDQ", &mycode, LIBXSMM_X86_INSTR_VPANDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDQ", &mycode, LIBXSMM_X86_INSTR_VPADDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDB", &mycode, LIBXSMM_X86_INSTR_VPADDB, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDW", &mycode, LIBXSMM_X86_INSTR_VPADDW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDD", &mycode, LIBXSMM_X86_INSTR_VPADDD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMADDWD", &mycode, LIBXSMM_X86_INSTR_VPMADDWD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMADDUBSW", &mycode, LIBXSMM_X86_INSTR_VPMADDUBSW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDSW", &mycode, LIBXSMM_X86_INSTR_VPADDSW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDSB", &mycode, LIBXSMM_X86_INSTR_VPADDSB, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSUBD", &mycode, LIBXSMM_X86_INSTR_VPSUBD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMAXSD", &mycode, LIBXSMM_X86_INSTR_VPMAXSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMAXSW", &mycode, LIBXSMM_X86_INSTR_VPMAXSW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMINSD", &mycode, LIBXSMM_X86_INSTR_VPMINSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_V4FMADDPS", &mycode, LIBXSMM_X86_INSTR_V4FMADDPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_V4FNMADDPS", &mycode, LIBXSMM_X86_INSTR_V4FNMADDPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_V4FMADDSS", &mycode, LIBXSMM_X86_INSTR_V4FMADDSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_V4FNMADDSS", &mycode, LIBXSMM_X86_INSTR_V4FNMADDSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VP4DPWSSDS", &mycode, LIBXSMM_X86_INSTR_VP4DPWSSDS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VP4DPWSSD", &mycode, LIBXSMM_X86_INSTR_VP4DPWSSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPDPBUSD", &mycode, LIBXSMM_X86_INSTR_VPDPBUSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPDPBUSDS", &mycode, LIBXSMM_X86_INSTR_VPDPBUSDS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPDPWSSD", &mycode, LIBXSMM_X86_INSTR_VPDPWSSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPDPWSSDS", &mycode, LIBXSMM_X86_INSTR_VPDPWSSDS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VDPBF16PS", &mycode, LIBXSMM_X86_INSTR_VDPBF16PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTNEPS2BF16", &mycode, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTNE2PS2BF16", &mycode, LIBXSMM_X86_INSTR_VCVTNE2PS2BF16, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMOVDQU64_LD", &mycode, LIBXSMM_X86_INSTR_VMOVDQU64_LD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMOVDQU64_ST", &mycode, LIBXSMM_X86_INSTR_VMOVDQU64_ST, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );

  /* AVX512 FP16 */
  test_evex_compute_mem_2reg_general( "evex_mem_VADDPH", &mycode, LIBXSMM_X86_INSTR_VADDPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VADDSH", &mycode, LIBXSMM_X86_INSTR_VADDSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCMPPH", &mycode, LIBXSMM_X86_INSTR_VCMPPH, 2, 0x01, 8, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCMPSH", &mycode, LIBXSMM_X86_INSTR_VCMPSH, 2, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VDIVPH", &mycode, LIBXSMM_X86_INSTR_VDIVPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VDIVSH", &mycode, LIBXSMM_X86_INSTR_VDIVSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFCMADDCPH", &mycode, LIBXSMM_X86_INSTR_VFCMADDCPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADDCPH", &mycode, LIBXSMM_X86_INSTR_VFMADDCPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFCMADDCSH", &mycode, LIBXSMM_X86_INSTR_VFCMADDCSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADDCSH", &mycode, LIBXSMM_X86_INSTR_VFMADDCSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFCMULCPH", &mycode, LIBXSMM_X86_INSTR_VFCMULCPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMULCPH", &mycode, LIBXSMM_X86_INSTR_VFMULCPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFCMULCSH", &mycode, LIBXSMM_X86_INSTR_VFCMULCSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMULCSH", &mycode, LIBXSMM_X86_INSTR_VFMULCSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADDSUB132PH", &mycode, LIBXSMM_X86_INSTR_VFMADDSUB132PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADDSUB213PH", &mycode, LIBXSMM_X86_INSTR_VFMADDSUB213PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADDSUB231PH", &mycode, LIBXSMM_X86_INSTR_VFMADDSUB231PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUBADD132PH", &mycode, LIBXSMM_X86_INSTR_VFMSUBADD132PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUBADD213PH", &mycode, LIBXSMM_X86_INSTR_VFMSUBADD213PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUBADD231PH", &mycode, LIBXSMM_X86_INSTR_VFMSUBADD231PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD132PH", &mycode, LIBXSMM_X86_INSTR_VFMADD132PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD213PH", &mycode, LIBXSMM_X86_INSTR_VFMADD213PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD231PH", &mycode, LIBXSMM_X86_INSTR_VFMADD231PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD132PH", &mycode, LIBXSMM_X86_INSTR_VFNMADD132PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD213PH", &mycode, LIBXSMM_X86_INSTR_VFNMADD213PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD231PH", &mycode, LIBXSMM_X86_INSTR_VFNMADD231PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD132SH", &mycode, LIBXSMM_X86_INSTR_VFMADD132SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD213SH", &mycode, LIBXSMM_X86_INSTR_VFMADD213SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD231SH", &mycode, LIBXSMM_X86_INSTR_VFMADD231SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD132SH", &mycode, LIBXSMM_X86_INSTR_VFNMADD132SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD213SH", &mycode, LIBXSMM_X86_INSTR_VFNMADD213SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD231SH", &mycode, LIBXSMM_X86_INSTR_VFNMADD231SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB132PH", &mycode, LIBXSMM_X86_INSTR_VFMSUB132PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB213PH", &mycode, LIBXSMM_X86_INSTR_VFMSUB213PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB231PH", &mycode, LIBXSMM_X86_INSTR_VFMSUB231PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB132PH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB213PH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB231PH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231PH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB132SH", &mycode, LIBXSMM_X86_INSTR_VFMSUB132SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB213SH", &mycode, LIBXSMM_X86_INSTR_VFMSUB213SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB231SH", &mycode, LIBXSMM_X86_INSTR_VFMSUB231SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB132SH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB213SH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB231SH", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231SH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCLASSPH", &mycode, LIBXSMM_X86_INSTR_VPCLASSPH, 1, 0x01, 8, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCLASSSH", &mycode, LIBXSMM_X86_INSTR_VPCLASSSH, 1, 0x01, 8, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VGETEXPPH", &mycode, LIBXSMM_X86_INSTR_VGETEXPPH, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VGETEXPSH", &mycode, LIBXSMM_X86_INSTR_VGETEXPSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VGETMANTPH", &mycode, LIBXSMM_X86_INSTR_VGETMANTPH, 1, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VGETMANTSH", &mycode, LIBXSMM_X86_INSTR_VGETMANTSH, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMAXPH", &mycode, LIBXSMM_X86_INSTR_VMAXPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMAXSH", &mycode, LIBXSMM_X86_INSTR_VMAXSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMINPH", &mycode, LIBXSMM_X86_INSTR_VMINPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMINSH", &mycode, LIBXSMM_X86_INSTR_VMINSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMOVSH_LD_MEM", &mycode, LIBXSMM_X86_INSTR_VMOVSH_LD_MEM, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMOVSH_ST_MEM", &mycode, LIBXSMM_X86_INSTR_VMOVSH_ST_MEM, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
#if 0
  test_evex_compute_mem_2reg_general( "evex_mem_VMOVW_LD", &mycode, LIBXSMM_X86_INSTR_VMOVW_LD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 1, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMOVW_ST", &mycode, LIBXSMM_X86_INSTR_VMOVW_ST, 1, LIBXSMM_X86_IMM_UNDEF, 32, 1, 1 );
#endif
  test_evex_compute_mem_2reg_general( "evex_mem_VMULPH", &mycode, LIBXSMM_X86_INSTR_VMULPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMULSH", &mycode, LIBXSMM_X86_INSTR_VMULSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRCPPH", &mycode, LIBXSMM_X86_INSTR_VRCPPH, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRCPSH", &mycode, LIBXSMM_X86_INSTR_VRCPSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VREDUCEPH", &mycode, LIBXSMM_X86_INSTR_VREDUCEPH, 1, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VREDUCESH", &mycode, LIBXSMM_X86_INSTR_VREDUCESH, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRNDSCALEPH", &mycode, LIBXSMM_X86_INSTR_VRNDSCALEPH, 1, 0x01, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRNDSCALESH", &mycode, LIBXSMM_X86_INSTR_VRNDSCALESH, 2, 0x01, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRSQRTPH", &mycode, LIBXSMM_X86_INSTR_VRSQRTPH, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRSQRTSH", &mycode, LIBXSMM_X86_INSTR_VRSQRTSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSCALEFPH", &mycode, LIBXSMM_X86_INSTR_VSCALEFPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSCALEFSH", &mycode, LIBXSMM_X86_INSTR_VSCALEFSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSQRTPH", &mycode, LIBXSMM_X86_INSTR_VSQRTPH, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSQRTSH", &mycode, LIBXSMM_X86_INSTR_VSQRTSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSUBPH", &mycode, LIBXSMM_X86_INSTR_VSUBPH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSUBSH", &mycode, LIBXSMM_X86_INSTR_VSUBSH, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0, 1 );

  /* testing AVX512 masking */
  test_mask_move( "mask_move_KMOVB_GPR_LD", &mycode, LIBXSMM_X86_INSTR_KMOVB_GPR_LD );
  test_mask_move( "mask_move_KMOVW_GPR_LD", &mycode, LIBXSMM_X86_INSTR_KMOVW_GPR_LD );
  test_mask_move( "mask_move_KMOVD_GPR_LD", &mycode, LIBXSMM_X86_INSTR_KMOVD_GPR_LD );
  test_mask_move( "mask_move_KMOVQ_GPR_LD", &mycode, LIBXSMM_X86_INSTR_KMOVQ_GPR_LD );
  test_mask_move( "mask_move_KMOVB_GPR_ST", &mycode, LIBXSMM_X86_INSTR_KMOVB_GPR_ST );
  test_mask_move( "mask_move_KMOVW_GPR_ST", &mycode, LIBXSMM_X86_INSTR_KMOVW_GPR_ST );
  test_mask_move( "mask_move_KMOVD_GPR_ST", &mycode, LIBXSMM_X86_INSTR_KMOVD_GPR_ST );
  test_mask_move( "mask_move_KMOVQ_GPR_ST", &mycode, LIBXSMM_X86_INSTR_KMOVQ_GPR_ST );
  test_mask_move_mem( "mask_move_KMOVB_LD", &mycode, LIBXSMM_X86_INSTR_KMOVB_LD );
  test_mask_move_mem( "mask_move_KMOVW_LD", &mycode, LIBXSMM_X86_INSTR_KMOVW_LD );
  test_mask_move_mem( "mask_move_KMOVD_LD", &mycode, LIBXSMM_X86_INSTR_KMOVD_LD );
  test_mask_move_mem( "mask_move_KMOVQ_LD", &mycode, LIBXSMM_X86_INSTR_KMOVQ_LD );
  test_mask_move_mem( "mask_move_KMOVB_ST", &mycode, LIBXSMM_X86_INSTR_KMOVB_ST );
  test_mask_move_mem( "mask_move_KMOVW_ST", &mycode, LIBXSMM_X86_INSTR_KMOVW_ST );
  test_mask_move_mem( "mask_move_KMOVD_ST", &mycode, LIBXSMM_X86_INSTR_KMOVD_ST );
  test_mask_move_mem( "mask_move_KMOVQ_ST", &mycode, LIBXSMM_X86_INSTR_KMOVQ_ST );
  test_mask_compute_reg( "mask_reg_KADDB", &mycode, LIBXSMM_X86_INSTR_KADDB, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KADDW", &mycode, LIBXSMM_X86_INSTR_KADDW, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KADDD", &mycode, LIBXSMM_X86_INSTR_KADDD, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KADDQ", &mycode, LIBXSMM_X86_INSTR_KADDQ, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KANDB", &mycode, LIBXSMM_X86_INSTR_KANDB, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KANDW", &mycode, LIBXSMM_X86_INSTR_KANDW, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KANDD", &mycode, LIBXSMM_X86_INSTR_KANDD, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KANDQ", &mycode, LIBXSMM_X86_INSTR_KANDQ, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KANDNB", &mycode, LIBXSMM_X86_INSTR_KANDNB, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KANDNW", &mycode, LIBXSMM_X86_INSTR_KANDNW, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KANDND", &mycode, LIBXSMM_X86_INSTR_KANDND, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KANDNQ", &mycode, LIBXSMM_X86_INSTR_KANDNQ, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KNOTB", &mycode, LIBXSMM_X86_INSTR_KNOTB, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KNOTW", &mycode, LIBXSMM_X86_INSTR_KNOTW, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KNOTD", &mycode, LIBXSMM_X86_INSTR_KNOTD, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KNOTQ", &mycode, LIBXSMM_X86_INSTR_KNOTQ, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KORB", &mycode, LIBXSMM_X86_INSTR_KORB, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KORW", &mycode, LIBXSMM_X86_INSTR_KORW, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KORD", &mycode, LIBXSMM_X86_INSTR_KORD, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KORQ", &mycode, LIBXSMM_X86_INSTR_KORQ, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KORTESTB", &mycode, LIBXSMM_X86_INSTR_KORTESTB, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KORTESTW", &mycode, LIBXSMM_X86_INSTR_KORTESTW, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KORTESTD", &mycode, LIBXSMM_X86_INSTR_KORTESTD, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KORTESTQ", &mycode, LIBXSMM_X86_INSTR_KORTESTQ, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KSHIFTLB", &mycode, LIBXSMM_X86_INSTR_KSHIFTLB, 1, 0x01 );
  test_mask_compute_reg( "mask_reg_KSHIFTLW", &mycode, LIBXSMM_X86_INSTR_KSHIFTLW, 1, 0x01 );
  test_mask_compute_reg( "mask_reg_KSHIFTLD", &mycode, LIBXSMM_X86_INSTR_KSHIFTLD, 1, 0x01 );
  test_mask_compute_reg( "mask_reg_KSHIFTLQ", &mycode, LIBXSMM_X86_INSTR_KSHIFTLQ, 1, 0x01 );
  test_mask_compute_reg( "mask_reg_KSHIFTRB", &mycode, LIBXSMM_X86_INSTR_KSHIFTRB, 1, 0x01 );
  test_mask_compute_reg( "mask_reg_KSHIFTRW", &mycode, LIBXSMM_X86_INSTR_KSHIFTRW, 1, 0x01 );
  test_mask_compute_reg( "mask_reg_KSHIFTRD", &mycode, LIBXSMM_X86_INSTR_KSHIFTRD, 1, 0x01 );
  test_mask_compute_reg( "mask_reg_KSHIFTRQ", &mycode, LIBXSMM_X86_INSTR_KSHIFTRQ, 1, 0x01 );
  test_mask_compute_reg( "mask_reg_KTESTB", &mycode, LIBXSMM_X86_INSTR_KTESTB, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KTESTW", &mycode, LIBXSMM_X86_INSTR_KTESTW, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KTESTD", &mycode, LIBXSMM_X86_INSTR_KTESTD, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KTESTQ", &mycode, LIBXSMM_X86_INSTR_KTESTQ, 1, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KUNPCKBW", &mycode, LIBXSMM_X86_INSTR_KUNPCKBW, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KUNPCKWD", &mycode, LIBXSMM_X86_INSTR_KUNPCKWD, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KUNPCKDQ", &mycode, LIBXSMM_X86_INSTR_KUNPCKDQ, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KXNORB", &mycode, LIBXSMM_X86_INSTR_KXNORB, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KXNORW", &mycode, LIBXSMM_X86_INSTR_KXNORW, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KXNORD", &mycode, LIBXSMM_X86_INSTR_KXNORD, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KXNORQ", &mycode, LIBXSMM_X86_INSTR_KXNORQ, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KXORB", &mycode, LIBXSMM_X86_INSTR_KXORB, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KXORW", &mycode, LIBXSMM_X86_INSTR_KXORW, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KXORD", &mycode, LIBXSMM_X86_INSTR_KXORD, 0, LIBXSMM_X86_IMM_UNDEF );
  test_mask_compute_reg( "mask_reg_KXORQ", &mycode, LIBXSMM_X86_INSTR_KXORQ, 0, LIBXSMM_X86_IMM_UNDEF );

  /* testing prefetches */
  test_prefetch( "pf_PREFETCHT0", &mycode, LIBXSMM_X86_INSTR_PREFETCHT0 );
  test_prefetch( "pf_PREFETCHT1", &mycode, LIBXSMM_X86_INSTR_PREFETCHT1 );
  test_prefetch( "pf_PREFETCHT2", &mycode, LIBXSMM_X86_INSTR_PREFETCHT2 );
  test_prefetch( "pf_PREFETCHNTA", &mycode, LIBXSMM_X86_INSTR_PREFETCHNTA );
  test_prefetch( "pf_PREFETCHW", &mycode, LIBXSMM_X86_INSTR_PREFETCHW );
  test_prefetch( "pf_CLDEMOTE", &mycode, LIBXSMM_X86_INSTR_CLDEMOTE );
  test_prefetch( "pf_CLFLUSH", &mycode, LIBXSMM_X86_INSTR_CLFLUSH );
  test_prefetch( "pf_CLFLUSHOPT", &mycode, LIBXSMM_X86_INSTR_CLFLUSHOPT );

  /* testing tile move */
  test_tile_move( "tile_mov_TILELOADD", &mycode, LIBXSMM_X86_INSTR_TILELOADD );
  test_tile_move( "tile_mov_TILELOADDT1", &mycode, LIBXSMM_X86_INSTR_TILELOADDT1 );
  test_tile_move( "tile_mov_TILESTORED", &mycode, LIBXSMM_X86_INSTR_TILESTORED );
  test_tile_move( "tile_mov_TILEZERO", &mycode, LIBXSMM_X86_INSTR_TILEZERO );

  /* testing tile compute */
  test_tile_compute( "tile_reg_TDPBSSD", &mycode, LIBXSMM_X86_INSTR_TDPBSSD );
  test_tile_compute( "tile_reg_TDPBSUD", &mycode, LIBXSMM_X86_INSTR_TDPBSUD );
  test_tile_compute( "tile_reg_TDPBUSD", &mycode, LIBXSMM_X86_INSTR_TDPBUSD );
  test_tile_compute( "tile_reg_TDPBUUD", &mycode, LIBXSMM_X86_INSTR_TDPBUUD );
  test_tile_compute( "tile_reg_TDPBF16PS", &mycode, LIBXSMM_X86_INSTR_TDPBF16PS );
  test_tile_compute( "tile_reg_TDPFP16PS", &mycode, LIBXSMM_X86_INSTR_TDPFP16PS );

  /* AVX only tests */
  mycode.arch = LIBXSMM_X86_AVX2_SRF;

  test_vex_load_store( "vex_mov_VMOVAPD", &mycode, LIBXSMM_X86_INSTR_VMOVAPD, 3 );
  test_vex_load_store( "vex_mov_VMOVAPS", &mycode, LIBXSMM_X86_INSTR_VMOVAPS, 3 );
  test_vex_load_store( "vex_mov_VMOVUPD", &mycode, LIBXSMM_X86_INSTR_VMOVUPD, 3 );
  test_vex_load_store( "vex_mov_VMOVUPS", &mycode, LIBXSMM_X86_INSTR_VMOVUPS, 3 );
  test_vex_load_store( "vex_mov_VMOVSS", &mycode, LIBXSMM_X86_INSTR_VMOVSS, 3 );
  test_vex_load_store( "vex_mov_VMOVSD", &mycode, LIBXSMM_X86_INSTR_VMOVSD, 3 );
  test_vex_load_store( "vex_mov_VBROADCASTSS", &mycode, LIBXSMM_X86_INSTR_VBROADCASTSS, 1 );
  test_vex_load_store( "vex_mov_VBROADCASTSD_VEX", &mycode, LIBXSMM_X86_INSTR_VBROADCASTSD_VEX, 1 );
  test_vex_load_store( "vex_mov_VPBROADCASTB", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTB, 1 );
  test_vex_load_store( "vex_mov_VPBROADCASTW", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTW, 1 );
  test_vex_load_store( "vex_mov_VPBROADCASTD", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTD, 1 );
  test_vex_load_store( "vex_mov_VPBROADCASTQ_VEX", &mycode, LIBXSMM_X86_INSTR_VPBROADCASTQ_VEX, 1 );
  test_vex_load_store( "vex_mov_VBROADCASTI128", &mycode, LIBXSMM_X86_INSTR_VBROADCASTI128, 1 );
  test_vex_load_store( "vex_mov_VMOVNTPD", &mycode, LIBXSMM_X86_INSTR_VMOVNTPD, 2 );
  test_vex_load_store( "vex_mov_VMOVNTPS", &mycode, LIBXSMM_X86_INSTR_VMOVNTPS, 2 );
  test_vex_load_store( "vex_mov_VBCSTNEBF162PS", &mycode, LIBXSMM_X86_INSTR_VBCSTNEBF162PS, 1 );
  test_vex_load_store( "vex_mov_VBCSTNESH2PS", &mycode, LIBXSMM_X86_INSTR_VBCSTNESH2PS, 1 );
  test_vex_load_store( "vex_mov_VCVTNEEBF162PS", &mycode, LIBXSMM_X86_INSTR_VCVTNEEBF162PS, 1 );
  test_vex_load_store( "vex_mov_VCVTNEEPH2PS", &mycode, LIBXSMM_X86_INSTR_VCVTNEEPH2PS, 1 );
  test_vex_load_store( "vex_mov_VCVTNEOBF162PS", &mycode, LIBXSMM_X86_INSTR_VCVTNEOBF162PS, 1 );
  test_vex_load_store( "vex_mov_VCVTNEOPH2PS", &mycode, LIBXSMM_X86_INSTR_VCVTNEOPH2PS, 1 );

  test_vex_mask_load_store( "vex_mov_VMASKMOVPD", &mycode, 0, LIBXSMM_X86_INSTR_VMASKMOVPD );
  test_vex_mask_load_store( "vex_mov_VMASKMOVPS", &mycode, 0, LIBXSMM_X86_INSTR_VMASKMOVPS );
  test_vex_mask_load_store( "vex_mov_VGATHERDPS_VEX", &mycode, 1, LIBXSMM_X86_INSTR_VGATHERDPS_VEX );
  test_vex_mask_load_store( "vex_mov_VGATHERDPD_VEX", &mycode, 1, LIBXSMM_X86_INSTR_VGATHERDPD_VEX );
  test_vex_mask_load_store( "vex_mov_VGATHERQPS_VEX", &mycode, 1, LIBXSMM_X86_INSTR_VGATHERQPS_VEX );
  test_vex_mask_load_store( "vex_mov_VGATHERQPD_VEX", &mycode, 1, LIBXSMM_X86_INSTR_VGATHERQPD_VEX );
  test_vex_mask_load_store( "vex_mov_VPGATHERDD_VEX", &mycode, 1, LIBXSMM_X86_INSTR_VPGATHERDD_VEX );
  test_vex_mask_load_store( "vex_mov_VPGATHERDQ_VEX", &mycode, 1, LIBXSMM_X86_INSTR_VPGATHERDQ_VEX );
  test_vex_mask_load_store( "vex_mov_VPGATHERQD_VEX", &mycode, 1, LIBXSMM_X86_INSTR_VPGATHERQD_VEX );
  test_vex_mask_load_store( "vex_mov_VPGATHERQQ_VEX", &mycode, 1, LIBXSMM_X86_INSTR_VPGATHERQQ_VEX );

  /* SSE1 tests */
  test_rex_vload_vstore( "rex_mov_MOVAPS", &mycode, LIBXSMM_X86_INSTR_MOVAPS, 3 );
  test_rex_vload_vstore( "rex_mov_MOVUPS", &mycode, LIBXSMM_X86_INSTR_MOVUPS, 3 );
  test_rex_vload_vstore( "rex_mov_MOVSS", &mycode, LIBXSMM_X86_INSTR_MOVSS, 3 );
  test_rex_vload_vstore( "rex_mov_MOVNTPS", &mycode, LIBXSMM_X86_INSTR_MOVNTPS, 2 );
  test_rex_vload_vstore( "rex_mov_MOVLPS", &mycode, LIBXSMM_X86_INSTR_MOVLPS, 1 );
  test_rex_vload_vstore( "rex_mov_MOVHPS", &mycode, LIBXSMM_X86_INSTR_MOVHPS, 1 );

  test_rex_vcompute_2reg_general( "rex_reg_MOVAPS_LD", &mycode, LIBXSMM_X86_INSTR_MOVAPS_LD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVAPS_ST", &mycode, LIBXSMM_X86_INSTR_MOVAPS_ST, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVUPS_LD", &mycode, LIBXSMM_X86_INSTR_MOVUPS_LD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVUPS_ST", &mycode, LIBXSMM_X86_INSTR_MOVUPS_ST, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVMSKPS", &mycode, LIBXSMM_X86_INSTR_MOVMSKPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ANDPS", &mycode, LIBXSMM_X86_INSTR_ANDPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ANDNPS", &mycode, LIBXSMM_X86_INSTR_ANDNPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ORPS", &mycode, LIBXSMM_X86_INSTR_ORPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_XORPS", &mycode, LIBXSMM_X86_INSTR_XORPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ADDPS", &mycode, LIBXSMM_X86_INSTR_ADDPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MULPS", &mycode, LIBXSMM_X86_INSTR_MULPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SUBPS", &mycode, LIBXSMM_X86_INSTR_SUBPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_DIVPS", &mycode, LIBXSMM_X86_INSTR_DIVPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_RCPPS", &mycode, LIBXSMM_X86_INSTR_RCPPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SQRTPS", &mycode, LIBXSMM_X86_INSTR_SQRTPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MAXPS", &mycode, LIBXSMM_X86_INSTR_MAXPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MINPS", &mycode, LIBXSMM_X86_INSTR_MINPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_RSQRTPS", &mycode, LIBXSMM_X86_INSTR_RSQRTPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CMPPS", &mycode, LIBXSMM_X86_INSTR_CMPPS, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_SHUFPS", &mycode, LIBXSMM_X86_INSTR_SHUFPS, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_UNPCKHPS", &mycode, LIBXSMM_X86_INSTR_UNPCKHPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_UNPCKLPS", &mycode, LIBXSMM_X86_INSTR_UNPCKLPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVSS_LD", &mycode, LIBXSMM_X86_INSTR_MOVSS_LD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVSS_ST", &mycode, LIBXSMM_X86_INSTR_MOVSS_ST, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ADDSS", &mycode, LIBXSMM_X86_INSTR_ADDSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MULSS", &mycode, LIBXSMM_X86_INSTR_MULSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SUBSS", &mycode, LIBXSMM_X86_INSTR_SUBSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_DIVSS", &mycode, LIBXSMM_X86_INSTR_DIVSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_RCPSS", &mycode, LIBXSMM_X86_INSTR_RCPSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SQRTSS", &mycode, LIBXSMM_X86_INSTR_SQRTSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MAXSS", &mycode, LIBXSMM_X86_INSTR_MAXSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MINSS", &mycode, LIBXSMM_X86_INSTR_MINSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_RSQRTSS", &mycode, LIBXSMM_X86_INSTR_RSQRTSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CMPSS", &mycode, LIBXSMM_X86_INSTR_CMPSS, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_COMISS", &mycode, LIBXSMM_X86_INSTR_COMISS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_UCOMISS", &mycode, LIBXSMM_X86_INSTR_UCOMISS, 0x0 );

  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVAPS_LD", &mycode, LIBXSMM_X86_INSTR_MOVAPS_LD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVAPS_ST", &mycode, LIBXSMM_X86_INSTR_MOVAPS_ST, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVUPS_LD", &mycode, LIBXSMM_X86_INSTR_MOVUPS_LD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVUPS_ST", &mycode, LIBXSMM_X86_INSTR_MOVUPS_ST, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ANDPS", &mycode, LIBXSMM_X86_INSTR_ANDPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ANDNPS", &mycode, LIBXSMM_X86_INSTR_ANDNPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ORPS", &mycode, LIBXSMM_X86_INSTR_ORPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_XORPS", &mycode, LIBXSMM_X86_INSTR_XORPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ADDPS", &mycode, LIBXSMM_X86_INSTR_ADDPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MULPS", &mycode, LIBXSMM_X86_INSTR_MULPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SUBPS", &mycode, LIBXSMM_X86_INSTR_SUBPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_DIVPS", &mycode, LIBXSMM_X86_INSTR_DIVPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_RCPPS", &mycode, LIBXSMM_X86_INSTR_RCPPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SQRTPS", &mycode, LIBXSMM_X86_INSTR_SQRTPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MAXPS", &mycode, LIBXSMM_X86_INSTR_MAXPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MINPS", &mycode, LIBXSMM_X86_INSTR_MINPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_RSQRTPS", &mycode, LIBXSMM_X86_INSTR_RSQRTPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CMPPS", &mycode, LIBXSMM_X86_INSTR_CMPPS, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SHUFPS", &mycode, LIBXSMM_X86_INSTR_SHUFPS, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_UNPCKHPS", &mycode, LIBXSMM_X86_INSTR_UNPCKHPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_UNPCKLPS", &mycode, LIBXSMM_X86_INSTR_UNPCKLPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVSS_LD", &mycode, LIBXSMM_X86_INSTR_MOVSS_LD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVSS_ST", &mycode, LIBXSMM_X86_INSTR_MOVSS_ST, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ADDSS", &mycode, LIBXSMM_X86_INSTR_ADDSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MULSS", &mycode, LIBXSMM_X86_INSTR_MULSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SUBSS", &mycode, LIBXSMM_X86_INSTR_SUBSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_DIVSS", &mycode, LIBXSMM_X86_INSTR_DIVSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_RCPSS", &mycode, LIBXSMM_X86_INSTR_RCPSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SQRTSS", &mycode, LIBXSMM_X86_INSTR_SQRTSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MAXSS", &mycode, LIBXSMM_X86_INSTR_MAXSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MINSS", &mycode, LIBXSMM_X86_INSTR_MINSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_RSQRTSS", &mycode, LIBXSMM_X86_INSTR_RSQRTSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CMPSS", &mycode, LIBXSMM_X86_INSTR_CMPSS, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_COMISS", &mycode, LIBXSMM_X86_INSTR_COMISS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_UCOMISS", &mycode, LIBXSMM_X86_INSTR_UCOMISS, 0x0 );

  /* SSE2 tests */
  test_rex_vload_vstore( "rex_mov_MOVAPD", &mycode, LIBXSMM_X86_INSTR_MOVAPD, 3 );
  test_rex_vload_vstore( "rex_mov_MOVUPD", &mycode, LIBXSMM_X86_INSTR_MOVUPD, 3 );
  test_rex_vload_vstore( "rex_mov_MOVSD", &mycode, LIBXSMM_X86_INSTR_MOVSD, 3 );
  test_rex_vload_vstore( "rex_mov_MOVNTPD", &mycode, LIBXSMM_X86_INSTR_MOVNTPD, 2 );
  test_rex_vload_vstore( "rex_mov_MOVDQA_LD", &mycode, LIBXSMM_X86_INSTR_MOVDQA_LD, 1 );
  test_rex_vload_vstore( "rex_mov_MOVDQA_ST", &mycode, LIBXSMM_X86_INSTR_MOVDQA_ST, 2 );
  test_rex_vload_vstore( "rex_mov_MOVDQU_LD", &mycode, LIBXSMM_X86_INSTR_MOVDQU_LD, 1 );
  test_rex_vload_vstore( "rex_mov_MOVDQU_ST", &mycode, LIBXSMM_X86_INSTR_MOVDQU_ST, 2 );
  test_rex_vload_vstore( "rex_mov_MOVNTDQ", &mycode, LIBXSMM_X86_INSTR_MOVNTDQ, 2 );

  test_rex_vcompute_2reg_general( "rex_reg_MOVD_SSE_LD", &mycode, LIBXSMM_X86_INSTR_MOVD_SSE_LD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVD_SSE_ST", &mycode, LIBXSMM_X86_INSTR_MOVD_SSE_ST, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVQ_SSE_LD", &mycode, LIBXSMM_X86_INSTR_MOVQ_SSE_LD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVQ_SSE_ST", &mycode, LIBXSMM_X86_INSTR_MOVQ_SSE_ST, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVAPD_LD", &mycode, LIBXSMM_X86_INSTR_MOVAPD_LD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVAPD_ST", &mycode, LIBXSMM_X86_INSTR_MOVAPD_ST, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVUPD_LD", &mycode, LIBXSMM_X86_INSTR_MOVUPD_LD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVUPD_ST", &mycode, LIBXSMM_X86_INSTR_MOVUPD_ST, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVMSKPD", &mycode, LIBXSMM_X86_INSTR_MOVMSKPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ANDPD", &mycode, LIBXSMM_X86_INSTR_ANDPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ANDNPD", &mycode, LIBXSMM_X86_INSTR_ANDNPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ORPD", &mycode, LIBXSMM_X86_INSTR_ORPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_XORPD", &mycode, LIBXSMM_X86_INSTR_XORPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ADDPD", &mycode, LIBXSMM_X86_INSTR_ADDPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MULPD", &mycode, LIBXSMM_X86_INSTR_MULPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SUBPD", &mycode, LIBXSMM_X86_INSTR_SUBPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_DIVPD", &mycode, LIBXSMM_X86_INSTR_DIVPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_RCPPD", &mycode, LIBXSMM_X86_INSTR_RCPPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SQRTPD", &mycode, LIBXSMM_X86_INSTR_SQRTPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MAXPD", &mycode, LIBXSMM_X86_INSTR_MAXPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MINPD", &mycode, LIBXSMM_X86_INSTR_MINPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_RSQRTPD", &mycode, LIBXSMM_X86_INSTR_RSQRTPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CMPPD", &mycode, LIBXSMM_X86_INSTR_CMPPD, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_SHUFPD", &mycode, LIBXSMM_X86_INSTR_SHUFPD, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_UNPCKHPD", &mycode, LIBXSMM_X86_INSTR_UNPCKHPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_UNPCKLPD", &mycode, LIBXSMM_X86_INSTR_UNPCKLPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVSD_LD", &mycode, LIBXSMM_X86_INSTR_MOVSD_LD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVSD_ST", &mycode, LIBXSMM_X86_INSTR_MOVSD_ST, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ADDSD", &mycode, LIBXSMM_X86_INSTR_ADDSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MULSD", &mycode, LIBXSMM_X86_INSTR_MULSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SUBSD", &mycode, LIBXSMM_X86_INSTR_SUBSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_DIVSD", &mycode, LIBXSMM_X86_INSTR_DIVSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_RCPSD", &mycode, LIBXSMM_X86_INSTR_RCPSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SQRTSD", &mycode, LIBXSMM_X86_INSTR_SQRTSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MAXSD", &mycode, LIBXSMM_X86_INSTR_MAXSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MINSD", &mycode, LIBXSMM_X86_INSTR_MINSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_RSQRTSD", &mycode, LIBXSMM_X86_INSTR_RSQRTSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CMPSD", &mycode, LIBXSMM_X86_INSTR_CMPSD, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_COMISD", &mycode, LIBXSMM_X86_INSTR_COMISD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_UCOMISD", &mycode, LIBXSMM_X86_INSTR_UCOMISD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVDQA_LD", &mycode, LIBXSMM_X86_INSTR_MOVDQA_LD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVDQA_ST", &mycode, LIBXSMM_X86_INSTR_MOVDQA_ST, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVDQU_LD", &mycode, LIBXSMM_X86_INSTR_MOVDQU_LD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MOVDQU_ST", &mycode, LIBXSMM_X86_INSTR_MOVDQU_ST, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PAND", &mycode, LIBXSMM_X86_INSTR_PAND, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PANDN", &mycode, LIBXSMM_X86_INSTR_PANDN, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_POR", &mycode, LIBXSMM_X86_INSTR_POR, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PXOR", &mycode, LIBXSMM_X86_INSTR_PXOR, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PACKSSWB", &mycode, LIBXSMM_X86_INSTR_PACKSSWB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PACKSSDW", &mycode, LIBXSMM_X86_INSTR_PACKSSDW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PACKUSWB", &mycode, LIBXSMM_X86_INSTR_PACKUSWB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PADDB", &mycode, LIBXSMM_X86_INSTR_PADDB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PADDW", &mycode, LIBXSMM_X86_INSTR_PADDW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PADDD", &mycode, LIBXSMM_X86_INSTR_PADDD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PADDQ", &mycode, LIBXSMM_X86_INSTR_PADDQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PADDSB", &mycode, LIBXSMM_X86_INSTR_PADDSB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PADDSW", &mycode, LIBXSMM_X86_INSTR_PADDSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PADDUSB", &mycode, LIBXSMM_X86_INSTR_PADDUSB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PADDUSW", &mycode, LIBXSMM_X86_INSTR_PADDUSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PAVGB", &mycode, LIBXSMM_X86_INSTR_PAVGB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PAVGW", &mycode, LIBXSMM_X86_INSTR_PAVGW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PCMPEQB", &mycode, LIBXSMM_X86_INSTR_PCMPEQB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PCMPEQW", &mycode, LIBXSMM_X86_INSTR_PCMPEQW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PCMPEQD", &mycode, LIBXSMM_X86_INSTR_PCMPEQD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PCMPGTB", &mycode, LIBXSMM_X86_INSTR_PCMPGTB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PCMPGTW", &mycode, LIBXSMM_X86_INSTR_PCMPGTW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PCMPGTD", &mycode, LIBXSMM_X86_INSTR_PCMPGTD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PEXTRW", &mycode, LIBXSMM_X86_INSTR_PEXTRW, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PINSRW", &mycode, LIBXSMM_X86_INSTR_PINSRW, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PMADDWD", &mycode, LIBXSMM_X86_INSTR_PMADDWD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMAXSW", &mycode, LIBXSMM_X86_INSTR_PMAXSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMAXUB", &mycode, LIBXSMM_X86_INSTR_PMAXUB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMINSW", &mycode, LIBXSMM_X86_INSTR_PMINSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMINUB", &mycode, LIBXSMM_X86_INSTR_PMINUB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVMSKB", &mycode, LIBXSMM_X86_INSTR_PMOVMSKB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMULHUW", &mycode, LIBXSMM_X86_INSTR_PMULHUW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMULHW", &mycode, LIBXSMM_X86_INSTR_PMULHW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMULLW", &mycode, LIBXSMM_X86_INSTR_PMULLW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMULUDQ", &mycode, LIBXSMM_X86_INSTR_PMULUDQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSADBW", &mycode, LIBXSMM_X86_INSTR_PSADBW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSHUFD", &mycode, LIBXSMM_X86_INSTR_PSHUFD, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PSHUFHW", &mycode, LIBXSMM_X86_INSTR_PSHUFHW, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PSHUFLW", &mycode, LIBXSMM_X86_INSTR_PSHUFLW, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PSLLW", &mycode, LIBXSMM_X86_INSTR_PSLLW, 0x0 );
  test_rex_vcompute_1reg_general( "rex_reg_PSLLW_I", &mycode, LIBXSMM_X86_INSTR_PSLLW_I, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PSLLD", &mycode, LIBXSMM_X86_INSTR_PSLLD, 0x0 );
  test_rex_vcompute_1reg_general( "rex_reg_PSLLD_I", &mycode, LIBXSMM_X86_INSTR_PSLLD_I, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PSLLQ", &mycode, LIBXSMM_X86_INSTR_PSLLQ, 0x0 );
  test_rex_vcompute_1reg_general( "rex_reg_PSLLQ_I", &mycode, LIBXSMM_X86_INSTR_PSLLQ_I, 0x2 );
  test_rex_vcompute_1reg_general( "rex_reg_PSLLDQ_I", &mycode, LIBXSMM_X86_INSTR_PSLLDQ_I, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PSRAW", &mycode, LIBXSMM_X86_INSTR_PSRAW, 0x0 );
  test_rex_vcompute_1reg_general( "rex_reg_PSRAW_I", &mycode, LIBXSMM_X86_INSTR_PSRAW_I, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PSRAD", &mycode, LIBXSMM_X86_INSTR_PSRAD, 0x0 );
  test_rex_vcompute_1reg_general( "rex_reg_PSRAD_I", &mycode, LIBXSMM_X86_INSTR_PSRAD_I, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PSRLW", &mycode, LIBXSMM_X86_INSTR_PSRLW, 0x0 );
  test_rex_vcompute_1reg_general( "rex_reg_PSRLW_I", &mycode, LIBXSMM_X86_INSTR_PSRLW_I, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PSRLD", &mycode, LIBXSMM_X86_INSTR_PSRLD, 0x0 );
  test_rex_vcompute_1reg_general( "rex_reg_PSRLD_I", &mycode, LIBXSMM_X86_INSTR_PSRLD_I, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PSRLQ", &mycode, LIBXSMM_X86_INSTR_PSRLQ, 0x0 );
  test_rex_vcompute_1reg_general( "rex_reg_PSRLQ_I", &mycode, LIBXSMM_X86_INSTR_PSRLQ_I, 0x2 );
  test_rex_vcompute_1reg_general( "rex_reg_PSRLDQ_I", &mycode, LIBXSMM_X86_INSTR_PSRLDQ_I, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PSUBB", &mycode, LIBXSMM_X86_INSTR_PSUBB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSUBW", &mycode, LIBXSMM_X86_INSTR_PSUBW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSUBD", &mycode, LIBXSMM_X86_INSTR_PSUBD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSUBQ", &mycode, LIBXSMM_X86_INSTR_PSUBQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSUBSB", &mycode, LIBXSMM_X86_INSTR_PSUBSB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSUBSW", &mycode, LIBXSMM_X86_INSTR_PSUBSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSUBUSB", &mycode, LIBXSMM_X86_INSTR_PSUBUSB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSUBUSW", &mycode, LIBXSMM_X86_INSTR_PSUBUSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PUNPCKHBW", &mycode, LIBXSMM_X86_INSTR_PUNPCKHBW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PUNPCKHWD", &mycode, LIBXSMM_X86_INSTR_PUNPCKHWD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PUNPCKHDQ", &mycode, LIBXSMM_X86_INSTR_PUNPCKHDQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PUNPCKHQDQ", &mycode, LIBXSMM_X86_INSTR_PUNPCKHQDQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PUNPCKLBW", &mycode, LIBXSMM_X86_INSTR_PUNPCKLBW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PUNPCKLWD", &mycode, LIBXSMM_X86_INSTR_PUNPCKLWD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PUNPCKLDQ", &mycode, LIBXSMM_X86_INSTR_PUNPCKLDQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PUNPCKLQDQ", &mycode, LIBXSMM_X86_INSTR_PUNPCKLQDQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CVTDQ2PD", &mycode, LIBXSMM_X86_INSTR_CVTDQ2PD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CVTDQ2PS", &mycode, LIBXSMM_X86_INSTR_CVTDQ2PS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CVTPD2DQ", &mycode, LIBXSMM_X86_INSTR_CVTPD2DQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CVTPD2PS", &mycode, LIBXSMM_X86_INSTR_CVTPD2PS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CVTPS2DQ", &mycode, LIBXSMM_X86_INSTR_CVTPS2DQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CVTPS2PD", &mycode, LIBXSMM_X86_INSTR_CVTPS2PD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CVTSD2SS", &mycode, LIBXSMM_X86_INSTR_CVTSD2SS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CVTSS2SD", &mycode, LIBXSMM_X86_INSTR_CVTSS2SD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CVTTPD2DQ", &mycode, LIBXSMM_X86_INSTR_CVTTPD2DQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_CVTTPS2DQ", &mycode, LIBXSMM_X86_INSTR_CVTTPS2DQ, 0x0 );

  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVD_SSE_LD", &mycode, LIBXSMM_X86_INSTR_MOVD_SSE_LD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVD_SSE_ST", &mycode, LIBXSMM_X86_INSTR_MOVD_SSE_ST, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVQ_SSE_LD", &mycode, LIBXSMM_X86_INSTR_MOVQ_SSE_LD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVQ_SSE_ST", &mycode, LIBXSMM_X86_INSTR_MOVQ_SSE_ST, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVAPD_LD", &mycode, LIBXSMM_X86_INSTR_MOVAPD_LD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVAPD_ST", &mycode, LIBXSMM_X86_INSTR_MOVAPD_ST, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVUPD_LD", &mycode, LIBXSMM_X86_INSTR_MOVUPD_LD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVUPD_ST", &mycode, LIBXSMM_X86_INSTR_MOVUPD_ST, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ANDPD", &mycode, LIBXSMM_X86_INSTR_ANDPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ANDNPD", &mycode, LIBXSMM_X86_INSTR_ANDNPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ORPD", &mycode, LIBXSMM_X86_INSTR_ORPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_XORPD", &mycode, LIBXSMM_X86_INSTR_XORPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ADDPD", &mycode, LIBXSMM_X86_INSTR_ADDPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MULPD", &mycode, LIBXSMM_X86_INSTR_MULPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SUBPD", &mycode, LIBXSMM_X86_INSTR_SUBPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_DIVPD", &mycode, LIBXSMM_X86_INSTR_DIVPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_RCPPD", &mycode, LIBXSMM_X86_INSTR_RCPPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SQRTPD", &mycode, LIBXSMM_X86_INSTR_SQRTPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MAXPD", &mycode, LIBXSMM_X86_INSTR_MAXPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MINPD", &mycode, LIBXSMM_X86_INSTR_MINPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_RSQRTPD", &mycode, LIBXSMM_X86_INSTR_RSQRTPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CMPPD", &mycode, LIBXSMM_X86_INSTR_CMPPD, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SHUFPD", &mycode, LIBXSMM_X86_INSTR_SHUFPD, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_UNPCKHPD", &mycode, LIBXSMM_X86_INSTR_UNPCKHPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_UNPCKLPD", &mycode, LIBXSMM_X86_INSTR_UNPCKLPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVSD_LD", &mycode, LIBXSMM_X86_INSTR_MOVSD_LD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVSD_ST", &mycode, LIBXSMM_X86_INSTR_MOVSD_ST, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ADDSD", &mycode, LIBXSMM_X86_INSTR_ADDSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MULSD", &mycode, LIBXSMM_X86_INSTR_MULSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SUBSD", &mycode, LIBXSMM_X86_INSTR_SUBSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_DIVSD", &mycode, LIBXSMM_X86_INSTR_DIVSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_RCPSD", &mycode, LIBXSMM_X86_INSTR_RCPSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SQRTSD", &mycode, LIBXSMM_X86_INSTR_SQRTSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MAXSD", &mycode, LIBXSMM_X86_INSTR_MAXSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MINSD", &mycode, LIBXSMM_X86_INSTR_MINSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_RSQRTSD", &mycode, LIBXSMM_X86_INSTR_RSQRTSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CMPSD", &mycode, LIBXSMM_X86_INSTR_CMPSD, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_COMISD", &mycode, LIBXSMM_X86_INSTR_COMISD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_UCOMISD", &mycode, LIBXSMM_X86_INSTR_UCOMISD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVDQA_LD", &mycode, LIBXSMM_X86_INSTR_MOVDQA_LD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVDQA_ST", &mycode, LIBXSMM_X86_INSTR_MOVDQA_ST, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVDQU_LD", &mycode, LIBXSMM_X86_INSTR_MOVDQU_LD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVDQU_ST", &mycode, LIBXSMM_X86_INSTR_MOVDQU_ST, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PAND", &mycode, LIBXSMM_X86_INSTR_PAND, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PANDN", &mycode, LIBXSMM_X86_INSTR_PANDN, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_POR", &mycode, LIBXSMM_X86_INSTR_POR, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PXOR", &mycode, LIBXSMM_X86_INSTR_PXOR, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PACKSSWB", &mycode, LIBXSMM_X86_INSTR_PACKSSWB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PACKSSDW", &mycode, LIBXSMM_X86_INSTR_PACKSSDW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PACKUSWB", &mycode, LIBXSMM_X86_INSTR_PACKUSWB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PADDB", &mycode, LIBXSMM_X86_INSTR_PADDB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PADDW", &mycode, LIBXSMM_X86_INSTR_PADDW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PADDD", &mycode, LIBXSMM_X86_INSTR_PADDD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PADDQ", &mycode, LIBXSMM_X86_INSTR_PADDQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PADDSB", &mycode, LIBXSMM_X86_INSTR_PADDSB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PADDSW", &mycode, LIBXSMM_X86_INSTR_PADDSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PADDUSB", &mycode, LIBXSMM_X86_INSTR_PADDUSB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PADDUSW", &mycode, LIBXSMM_X86_INSTR_PADDUSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PAVGB", &mycode, LIBXSMM_X86_INSTR_PAVGB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PAVGW", &mycode, LIBXSMM_X86_INSTR_PAVGW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PCMPEQB", &mycode, LIBXSMM_X86_INSTR_PCMPEQB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PCMPEQW", &mycode, LIBXSMM_X86_INSTR_PCMPEQW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PCMPEQD", &mycode, LIBXSMM_X86_INSTR_PCMPEQD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PCMPGTB", &mycode, LIBXSMM_X86_INSTR_PCMPGTB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PCMPGTW", &mycode, LIBXSMM_X86_INSTR_PCMPGTW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PCMPGTD", &mycode, LIBXSMM_X86_INSTR_PCMPGTD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PEXTRW", &mycode, LIBXSMM_X86_INSTR_PEXTRW, 0x2 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PINSRW", &mycode, LIBXSMM_X86_INSTR_PINSRW, 0x2 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMADDWD", &mycode, LIBXSMM_X86_INSTR_PMADDWD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMAXSW", &mycode, LIBXSMM_X86_INSTR_PMAXSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMAXUB", &mycode, LIBXSMM_X86_INSTR_PMAXUB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMINSW", &mycode, LIBXSMM_X86_INSTR_PMINSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMINUB", &mycode, LIBXSMM_X86_INSTR_PMINUB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMULHUW", &mycode, LIBXSMM_X86_INSTR_PMULHUW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMULHW", &mycode, LIBXSMM_X86_INSTR_PMULHW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMULLW", &mycode, LIBXSMM_X86_INSTR_PMULLW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMULUDQ", &mycode, LIBXSMM_X86_INSTR_PMULUDQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSADBW", &mycode, LIBXSMM_X86_INSTR_PSADBW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSHUFD", &mycode, LIBXSMM_X86_INSTR_PSHUFD, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSHUFHW", &mycode, LIBXSMM_X86_INSTR_PSHUFHW, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSHUFLW", &mycode, LIBXSMM_X86_INSTR_PSHUFLW, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSLLW", &mycode, LIBXSMM_X86_INSTR_PSLLW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSLLD", &mycode, LIBXSMM_X86_INSTR_PSLLD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSLLQ", &mycode, LIBXSMM_X86_INSTR_PSLLQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSRAW", &mycode, LIBXSMM_X86_INSTR_PSRAW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSRAD", &mycode, LIBXSMM_X86_INSTR_PSRAD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSRLW", &mycode, LIBXSMM_X86_INSTR_PSRLW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSRLD", &mycode, LIBXSMM_X86_INSTR_PSRLD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSRLQ", &mycode, LIBXSMM_X86_INSTR_PSRLQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSUBB", &mycode, LIBXSMM_X86_INSTR_PSUBB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSUBW", &mycode, LIBXSMM_X86_INSTR_PSUBW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSUBD", &mycode, LIBXSMM_X86_INSTR_PSUBD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSUBQ", &mycode, LIBXSMM_X86_INSTR_PSUBQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSUBSB", &mycode, LIBXSMM_X86_INSTR_PSUBSB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSUBSW", &mycode, LIBXSMM_X86_INSTR_PSUBSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSUBUSB", &mycode, LIBXSMM_X86_INSTR_PSUBUSB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSUBUSW", &mycode, LIBXSMM_X86_INSTR_PSUBUSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PUNPCKHBW", &mycode, LIBXSMM_X86_INSTR_PUNPCKHBW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PUNPCKHWD", &mycode, LIBXSMM_X86_INSTR_PUNPCKHWD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PUNPCKHDQ", &mycode, LIBXSMM_X86_INSTR_PUNPCKHDQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PUNPCKHQDQ", &mycode, LIBXSMM_X86_INSTR_PUNPCKHQDQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PUNPCKLBW", &mycode, LIBXSMM_X86_INSTR_PUNPCKLBW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PUNPCKLWD", &mycode, LIBXSMM_X86_INSTR_PUNPCKLWD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PUNPCKLDQ", &mycode, LIBXSMM_X86_INSTR_PUNPCKLDQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PUNPCKLQDQ", &mycode, LIBXSMM_X86_INSTR_PUNPCKLQDQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CVTDQ2PD", &mycode, LIBXSMM_X86_INSTR_CVTDQ2PD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CVTDQ2PS", &mycode, LIBXSMM_X86_INSTR_CVTDQ2PS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CVTPD2DQ", &mycode, LIBXSMM_X86_INSTR_CVTPD2DQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CVTPD2PS", &mycode, LIBXSMM_X86_INSTR_CVTPD2PS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CVTPS2DQ", &mycode, LIBXSMM_X86_INSTR_CVTPS2DQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CVTPS2PD", &mycode, LIBXSMM_X86_INSTR_CVTPS2PD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CVTSD2SS", &mycode, LIBXSMM_X86_INSTR_CVTSD2SS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CVTSS2SD", &mycode, LIBXSMM_X86_INSTR_CVTSS2SD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CVTTPD2DQ", &mycode, LIBXSMM_X86_INSTR_CVTTPD2DQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_CVTTPS2DQ", &mycode, LIBXSMM_X86_INSTR_CVTTPS2DQ, 0x0 );

  /* SSE3 tests */
  test_rex_vload_vstore( "rex_mov_LDDQU", &mycode, LIBXSMM_X86_INSTR_LDDQU, 1 );
  test_rex_vload_vstore( "rex_mov_MOVDDUP", &mycode, LIBXSMM_X86_INSTR_MOVDDUP, 1 );
  test_rex_vload_vstore( "rex_mov_MOVSHDUP", &mycode, LIBXSMM_X86_INSTR_MOVSHDUP, 1 );
  test_rex_vload_vstore( "rex_mov_MOVSLDUP", &mycode, LIBXSMM_X86_INSTR_MOVSLDUP, 1 );

  test_rex_vcompute_2reg_general( "rex_reg_ADDSUBPD", &mycode, LIBXSMM_X86_INSTR_ADDSUBPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ADDSUBPS", &mycode, LIBXSMM_X86_INSTR_ADDSUBPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_HADDPD", &mycode, LIBXSMM_X86_INSTR_HADDPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_HADDPS", &mycode, LIBXSMM_X86_INSTR_HADDPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_HSUBPD", &mycode, LIBXSMM_X86_INSTR_HSUBPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_HSUBPS", &mycode, LIBXSMM_X86_INSTR_HSUBPS, 0x0 );

  test_rex_vcompute_mem_1reg_general( "rex_mem_LDDQU", &mycode, LIBXSMM_X86_INSTR_LDDQU, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVDDUP", &mycode, LIBXSMM_X86_INSTR_MOVDDUP, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVSHDUP", &mycode, LIBXSMM_X86_INSTR_MOVSHDUP, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVSLDUP", &mycode, LIBXSMM_X86_INSTR_MOVSLDUP, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ADDSUBPD", &mycode, LIBXSMM_X86_INSTR_ADDSUBPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ADDSUBPS", &mycode, LIBXSMM_X86_INSTR_ADDSUBPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_HADDPD", &mycode, LIBXSMM_X86_INSTR_HADDPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_HADDPS", &mycode, LIBXSMM_X86_INSTR_HADDPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_HSUBPD", &mycode, LIBXSMM_X86_INSTR_HSUBPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_HSUBPS", &mycode, LIBXSMM_X86_INSTR_HSUBPS, 0x0 );

  /* SSSE3 tests */
  test_rex_vcompute_2reg_general( "rex_reg_PABSB", &mycode, LIBXSMM_X86_INSTR_PABSB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PABSW", &mycode, LIBXSMM_X86_INSTR_PABSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PABSD", &mycode, LIBXSMM_X86_INSTR_PABSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PALIGNR", &mycode, LIBXSMM_X86_INSTR_PALIGNR, 0x2 );
  test_rex_vcompute_2reg_general( "rex_reg_PHADDW", &mycode, LIBXSMM_X86_INSTR_PHADDW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PHADDD", &mycode, LIBXSMM_X86_INSTR_PHADDD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PHADDSW", &mycode, LIBXSMM_X86_INSTR_PHADDSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PHSUBW", &mycode, LIBXSMM_X86_INSTR_PHSUBW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PHSUBD", &mycode, LIBXSMM_X86_INSTR_PHSUBD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PHSUBSW", &mycode, LIBXSMM_X86_INSTR_PHSUBSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMADDUBSW", &mycode, LIBXSMM_X86_INSTR_PMADDUBSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMULHRSW", &mycode, LIBXSMM_X86_INSTR_PMULHRSW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSHUFB", &mycode, LIBXSMM_X86_INSTR_PSHUFB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSIGNB", &mycode, LIBXSMM_X86_INSTR_PSIGNB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSIGNW", &mycode, LIBXSMM_X86_INSTR_PSIGNW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PSIGND", &mycode, LIBXSMM_X86_INSTR_PSIGND, 0x0 );

  test_rex_vcompute_mem_1reg_general( "rex_mem_PABSB", &mycode, LIBXSMM_X86_INSTR_PABSB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PABSW", &mycode, LIBXSMM_X86_INSTR_PABSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PABSD", &mycode, LIBXSMM_X86_INSTR_PABSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PALIGNR", &mycode, LIBXSMM_X86_INSTR_PALIGNR, 0x2 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PHADDW", &mycode, LIBXSMM_X86_INSTR_PHADDW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PHADDD", &mycode, LIBXSMM_X86_INSTR_PHADDD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PHADDSW", &mycode, LIBXSMM_X86_INSTR_PHADDSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PHSUBW", &mycode, LIBXSMM_X86_INSTR_PHSUBW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PHSUBD", &mycode, LIBXSMM_X86_INSTR_PHSUBD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PHSUBSW", &mycode, LIBXSMM_X86_INSTR_PHSUBSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMADDUBSW", &mycode, LIBXSMM_X86_INSTR_PMADDUBSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMULHRSW", &mycode, LIBXSMM_X86_INSTR_PMULHRSW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSHUFB", &mycode, LIBXSMM_X86_INSTR_PSHUFB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSIGNB", &mycode, LIBXSMM_X86_INSTR_PSIGNB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSIGNW", &mycode, LIBXSMM_X86_INSTR_PSIGNW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PSIGND", &mycode, LIBXSMM_X86_INSTR_PSIGND, 0x0 );

  /* SSE4.1 instructions */
  test_rex_vload_vstore( "rex_mov_PMOVSXBW", &mycode, LIBXSMM_X86_INSTR_PMOVSXBW, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVSXBD", &mycode, LIBXSMM_X86_INSTR_PMOVSXBD, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVSXBQ", &mycode, LIBXSMM_X86_INSTR_PMOVSXBQ, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVSXWD", &mycode, LIBXSMM_X86_INSTR_PMOVSXWD, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVSXWQ", &mycode, LIBXSMM_X86_INSTR_PMOVSXWQ, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVSXDQ", &mycode, LIBXSMM_X86_INSTR_PMOVSXDQ, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVZXBW", &mycode, LIBXSMM_X86_INSTR_PMOVZXBW, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVZXBD", &mycode, LIBXSMM_X86_INSTR_PMOVZXBD, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVZXBQ", &mycode, LIBXSMM_X86_INSTR_PMOVZXBQ, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVZXWD", &mycode, LIBXSMM_X86_INSTR_PMOVZXWD, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVZXWQ", &mycode, LIBXSMM_X86_INSTR_PMOVZXWQ, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVZXDQ", &mycode, LIBXSMM_X86_INSTR_PMOVZXDQ, 1 );
  test_rex_vload_vstore( "rex_mov_PMOVZXDQ", &mycode, LIBXSMM_X86_INSTR_PMOVZXDQ, 1 );
  test_rex_vload_vstore( "rex_mov_MOVNTDQA", &mycode, LIBXSMM_X86_INSTR_MOVNTDQA, 1 );

  test_rex_vcompute_2reg_general( "rex_reg_BLENDPD", &mycode, LIBXSMM_X86_INSTR_BLENDPD, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_BLENDPS", &mycode, LIBXSMM_X86_INSTR_BLENDPS, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_BLENDVPD", &mycode, LIBXSMM_X86_INSTR_BLENDVPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_BLENDVPS", &mycode, LIBXSMM_X86_INSTR_BLENDVPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_DPPD", &mycode, LIBXSMM_X86_INSTR_DPPD, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_DPPS", &mycode, LIBXSMM_X86_INSTR_DPPS, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_EXTRACTPS", &mycode, LIBXSMM_X86_INSTR_EXTRACTPS, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_INSERTPS", &mycode, LIBXSMM_X86_INSTR_INSERTPS, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_ROUNDPD", &mycode, LIBXSMM_X86_INSTR_ROUNDPD, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_ROUNDPS", &mycode, LIBXSMM_X86_INSTR_ROUNDPS, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_ROUNDSD", &mycode, LIBXSMM_X86_INSTR_ROUNDSD, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_ROUNDSS", &mycode, LIBXSMM_X86_INSTR_ROUNDSS, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PBLENDW", &mycode, LIBXSMM_X86_INSTR_PBLENDW, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PBLENDVB", &mycode, LIBXSMM_X86_INSTR_PBLENDVB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PCMPEQQ", &mycode, LIBXSMM_X86_INSTR_PCMPEQQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVSXBW", &mycode, LIBXSMM_X86_INSTR_PMOVSXBW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVSXBD", &mycode, LIBXSMM_X86_INSTR_PMOVSXBD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVSXBQ", &mycode, LIBXSMM_X86_INSTR_PMOVSXBQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVSXWD", &mycode, LIBXSMM_X86_INSTR_PMOVSXWD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVSXWQ", &mycode, LIBXSMM_X86_INSTR_PMOVSXWQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVSXDQ", &mycode, LIBXSMM_X86_INSTR_PMOVSXDQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVZXBW", &mycode, LIBXSMM_X86_INSTR_PMOVZXBW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVZXBD", &mycode, LIBXSMM_X86_INSTR_PMOVZXBD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVZXBQ", &mycode, LIBXSMM_X86_INSTR_PMOVZXBQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVZXWD", &mycode, LIBXSMM_X86_INSTR_PMOVZXWD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVZXWQ", &mycode, LIBXSMM_X86_INSTR_PMOVZXWQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMOVZXDQ", &mycode, LIBXSMM_X86_INSTR_PMOVZXDQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PEXTRB", &mycode, LIBXSMM_X86_INSTR_PEXTRB, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PEXTRD", &mycode, LIBXSMM_X86_INSTR_PEXTRD, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PEXTRQ", &mycode, LIBXSMM_X86_INSTR_PEXTRQ, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PHMINPOSUW", &mycode, LIBXSMM_X86_INSTR_PHMINPOSUW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PINSRB", &mycode, LIBXSMM_X86_INSTR_PINSRB, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PINSRD", &mycode, LIBXSMM_X86_INSTR_PINSRD, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PINSRQ", &mycode, LIBXSMM_X86_INSTR_PINSRQ, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PMAXSB", &mycode, LIBXSMM_X86_INSTR_PMAXSB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMAXSD", &mycode, LIBXSMM_X86_INSTR_PMAXSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMAXUW", &mycode, LIBXSMM_X86_INSTR_PMAXUW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMAXUD", &mycode, LIBXSMM_X86_INSTR_PMAXUD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMINSB", &mycode, LIBXSMM_X86_INSTR_PMINSB, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMINSD", &mycode, LIBXSMM_X86_INSTR_PMINSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMINUW", &mycode, LIBXSMM_X86_INSTR_PMINUW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMINUD", &mycode, LIBXSMM_X86_INSTR_PMINUD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MPSADBW", &mycode, LIBXSMM_X86_INSTR_MPSADBW, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_PMULDQ", &mycode, LIBXSMM_X86_INSTR_PMULDQ, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PMULLD", &mycode, LIBXSMM_X86_INSTR_PMULLD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PACKUSDW", &mycode, LIBXSMM_X86_INSTR_PACKUSDW, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_PTEST", &mycode, LIBXSMM_X86_INSTR_PTEST, 0x0 );

  test_rex_vcompute_mem_1reg_general( "rex_mem_MOVNTDQA", &mycode, LIBXSMM_X86_INSTR_MOVNTDQA, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_BLENDPD", &mycode, LIBXSMM_X86_INSTR_BLENDPD, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_BLENDPS", &mycode, LIBXSMM_X86_INSTR_BLENDPS, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_BLENDVPD", &mycode, LIBXSMM_X86_INSTR_BLENDVPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_BLENDVPS", &mycode, LIBXSMM_X86_INSTR_BLENDVPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_DPPD", &mycode, LIBXSMM_X86_INSTR_DPPD, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_DPPS", &mycode, LIBXSMM_X86_INSTR_DPPS, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_EXTRACTPS", &mycode, LIBXSMM_X86_INSTR_EXTRACTPS, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_INSERTPS", &mycode, LIBXSMM_X86_INSTR_INSERTPS, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ROUNDPD", &mycode, LIBXSMM_X86_INSTR_ROUNDPD, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ROUNDPS", &mycode, LIBXSMM_X86_INSTR_ROUNDPS, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ROUNDSD", &mycode, LIBXSMM_X86_INSTR_ROUNDSD, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ROUNDSS", &mycode, LIBXSMM_X86_INSTR_ROUNDSS, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PBLENDW", &mycode, LIBXSMM_X86_INSTR_PBLENDW, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PBLENDVB", &mycode, LIBXSMM_X86_INSTR_PBLENDVB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PCMPEQQ", &mycode, LIBXSMM_X86_INSTR_PCMPEQQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVSXBW", &mycode, LIBXSMM_X86_INSTR_PMOVSXBW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVSXBD", &mycode, LIBXSMM_X86_INSTR_PMOVSXBW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVSXBQ", &mycode, LIBXSMM_X86_INSTR_PMOVSXBQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVSXWD", &mycode, LIBXSMM_X86_INSTR_PMOVSXWD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVSXWQ", &mycode, LIBXSMM_X86_INSTR_PMOVSXWQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVSXDQ", &mycode, LIBXSMM_X86_INSTR_PMOVSXDQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVZXBW", &mycode, LIBXSMM_X86_INSTR_PMOVZXBW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVZXBD", &mycode, LIBXSMM_X86_INSTR_PMOVZXBW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVZXBQ", &mycode, LIBXSMM_X86_INSTR_PMOVZXBQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVZXWD", &mycode, LIBXSMM_X86_INSTR_PMOVZXWD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVZXWQ", &mycode, LIBXSMM_X86_INSTR_PMOVZXWQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMOVZXDQ", &mycode, LIBXSMM_X86_INSTR_PMOVZXDQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PEXTRB", &mycode, LIBXSMM_X86_INSTR_PEXTRB, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PEXTRD", &mycode, LIBXSMM_X86_INSTR_PEXTRD, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PEXTRQ", &mycode, LIBXSMM_X86_INSTR_PEXTRQ, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PHMINPOSUW", &mycode, LIBXSMM_X86_INSTR_PHMINPOSUW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PINSRB", &mycode, LIBXSMM_X86_INSTR_PINSRB, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PINSRD", &mycode, LIBXSMM_X86_INSTR_PINSRD, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PINSRQ", &mycode, LIBXSMM_X86_INSTR_PINSRQ, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMAXSB", &mycode, LIBXSMM_X86_INSTR_PMAXSB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMAXSD", &mycode, LIBXSMM_X86_INSTR_PMAXSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMAXUW", &mycode, LIBXSMM_X86_INSTR_PMAXUW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMAXUD", &mycode, LIBXSMM_X86_INSTR_PMAXUD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMINSB", &mycode, LIBXSMM_X86_INSTR_PMINSB, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMINSD", &mycode, LIBXSMM_X86_INSTR_PMINSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMINUW", &mycode, LIBXSMM_X86_INSTR_PMINUW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMINUD", &mycode, LIBXSMM_X86_INSTR_PMINUD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MPSADBW", &mycode, LIBXSMM_X86_INSTR_MPSADBW, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMULDQ", &mycode, LIBXSMM_X86_INSTR_PMULDQ, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PMULLD", &mycode, LIBXSMM_X86_INSTR_PMULLD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PACKUSDW", &mycode, LIBXSMM_X86_INSTR_PACKUSDW, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_PTEST", &mycode, LIBXSMM_X86_INSTR_PTEST, 0x0 );

  /* SSE4.2 instructions */
  test_rex_vcompute_2reg_general( "rex_reg_PCMPGTQ", &mycode, LIBXSMM_X86_INSTR_PCMPGTQ, 0x0 );

  test_rex_vcompute_mem_1reg_general( "rex_mem_PCMPGTQ", &mycode, LIBXSMM_X86_INSTR_PCMPGTQ, 0x0 );

  /* test VEX/GP instructions */
  test_alu_reg( "alu_reg_ADDQ", &mycode, LIBXSMM_X86_INSTR_ADDQ, 0 );
  test_alu_reg( "alu_reg_ADDB_RM_R", &mycode, LIBXSMM_X86_INSTR_ADDB_RM_R, 0 );
  test_alu_reg( "alu_reg_ADDW_RM_R", &mycode, LIBXSMM_X86_INSTR_ADDW_RM_R, 0 );
  test_alu_reg( "alu_reg_ADDD_RM_R", &mycode, LIBXSMM_X86_INSTR_ADDD_RM_R, 0 );
  test_alu_reg( "alu_reg_ADDQ_RM_R", &mycode, LIBXSMM_X86_INSTR_ADDQ_RM_R, 0 );
  test_alu_reg( "alu_reg_ADDB_R_RM", &mycode, LIBXSMM_X86_INSTR_ADDB_R_RM, 0 );
  test_alu_reg( "alu_reg_ADDW_R_RM", &mycode, LIBXSMM_X86_INSTR_ADDW_R_RM, 0 );
  test_alu_reg( "alu_reg_ADDD_R_RM", &mycode, LIBXSMM_X86_INSTR_ADDD_R_RM, 0 );
  test_alu_reg( "alu_reg_ADDQ_R_RM", &mycode, LIBXSMM_X86_INSTR_ADDQ_R_RM, 0 );
  test_alu_reg( "alu_reg_ANDQ", &mycode, LIBXSMM_X86_INSTR_ANDQ, 0 );
  test_alu_reg( "alu_reg_ANDB_RM_R", &mycode, LIBXSMM_X86_INSTR_ANDB_RM_R, 0 );
  test_alu_reg( "alu_reg_ANDW_RM_R", &mycode, LIBXSMM_X86_INSTR_ANDW_RM_R, 0 );
  test_alu_reg( "alu_reg_ANDD_RM_R", &mycode, LIBXSMM_X86_INSTR_ANDD_RM_R, 0 );
  test_alu_reg( "alu_reg_ANDQ_RM_R", &mycode, LIBXSMM_X86_INSTR_ANDQ_RM_R, 0 );
  test_alu_reg( "alu_reg_ANDB_R_RM", &mycode, LIBXSMM_X86_INSTR_ANDB_R_RM, 0 );
  test_alu_reg( "alu_reg_ANDW_R_RM", &mycode, LIBXSMM_X86_INSTR_ANDW_R_RM, 0 );
  test_alu_reg( "alu_reg_ANDD_R_RM", &mycode, LIBXSMM_X86_INSTR_ANDD_R_RM, 0 );
  test_alu_reg( "alu_reg_ANDQ_R_RM", &mycode, LIBXSMM_X86_INSTR_ANDQ_R_RM, 0 );
  test_alu_reg( "alu_reg_CMOVAW", &mycode, LIBXSMM_X86_INSTR_CMOVAW, 0 );
  test_alu_reg( "alu_reg_CMOVAD", &mycode, LIBXSMM_X86_INSTR_CMOVAD, 0 );
  test_alu_reg( "alu_reg_CMOVAQ", &mycode, LIBXSMM_X86_INSTR_CMOVAQ, 0 );
  test_alu_reg( "alu_reg_CMOVAEW", &mycode, LIBXSMM_X86_INSTR_CMOVAEW, 0 );
  test_alu_reg( "alu_reg_CMOVAED", &mycode, LIBXSMM_X86_INSTR_CMOVAED, 0 );
  test_alu_reg( "alu_reg_CMOVAEQ", &mycode, LIBXSMM_X86_INSTR_CMOVAEQ, 0 );
  test_alu_reg( "alu_reg_CMOVBW", &mycode, LIBXSMM_X86_INSTR_CMOVBW, 0 );
  test_alu_reg( "alu_reg_CMOVBD", &mycode, LIBXSMM_X86_INSTR_CMOVBD, 0 );
  test_alu_reg( "alu_reg_CMOVBQ", &mycode, LIBXSMM_X86_INSTR_CMOVBQ, 0 );
  test_alu_reg( "alu_reg_CMOVBEW", &mycode, LIBXSMM_X86_INSTR_CMOVBEW, 0 );
  test_alu_reg( "alu_reg_CMOVBED", &mycode, LIBXSMM_X86_INSTR_CMOVBED, 0 );
  test_alu_reg( "alu_reg_CMOVBEQ", &mycode, LIBXSMM_X86_INSTR_CMOVBEQ, 0 );
  test_alu_reg( "alu_reg_CMOVCW", &mycode, LIBXSMM_X86_INSTR_CMOVCW, 0 );
  test_alu_reg( "alu_reg_CMOVCD", &mycode, LIBXSMM_X86_INSTR_CMOVCD, 0 );
  test_alu_reg( "alu_reg_CMOVCQ", &mycode, LIBXSMM_X86_INSTR_CMOVCQ, 0 );
  test_alu_reg( "alu_reg_CMOVEW", &mycode, LIBXSMM_X86_INSTR_CMOVEW, 0 );
  test_alu_reg( "alu_reg_CMOVED", &mycode, LIBXSMM_X86_INSTR_CMOVED, 0 );
  test_alu_reg( "alu_reg_CMOVEQ", &mycode, LIBXSMM_X86_INSTR_CMOVEQ, 0 );
  test_alu_reg( "alu_reg_CMOVGW", &mycode, LIBXSMM_X86_INSTR_CMOVGW, 0 );
  test_alu_reg( "alu_reg_CMOVGD", &mycode, LIBXSMM_X86_INSTR_CMOVGD, 0 );
  test_alu_reg( "alu_reg_CMOVGQ", &mycode, LIBXSMM_X86_INSTR_CMOVGQ, 0 );
  test_alu_reg( "alu_reg_CMOVGEW", &mycode, LIBXSMM_X86_INSTR_CMOVGEW, 0 );
  test_alu_reg( "alu_reg_CMOVGED", &mycode, LIBXSMM_X86_INSTR_CMOVGED, 0 );
  test_alu_reg( "alu_reg_CMOVGEQ", &mycode, LIBXSMM_X86_INSTR_CMOVGEQ, 0 );
  test_alu_reg( "alu_reg_CMOVLW", &mycode, LIBXSMM_X86_INSTR_CMOVLW, 0 );
  test_alu_reg( "alu_reg_CMOVLD", &mycode, LIBXSMM_X86_INSTR_CMOVLD, 0 );
  test_alu_reg( "alu_reg_CMOVLQ", &mycode, LIBXSMM_X86_INSTR_CMOVLQ, 0 );
  test_alu_reg( "alu_reg_CMOVLEW", &mycode, LIBXSMM_X86_INSTR_CMOVLEW, 0 );
  test_alu_reg( "alu_reg_CMOVLED", &mycode, LIBXSMM_X86_INSTR_CMOVLED, 0 );
  test_alu_reg( "alu_reg_CMOVLEQ", &mycode, LIBXSMM_X86_INSTR_CMOVLEQ, 0 );
  test_alu_reg( "alu_reg_CMOVNAW", &mycode, LIBXSMM_X86_INSTR_CMOVNAW, 0 );
  test_alu_reg( "alu_reg_CMOVNAD", &mycode, LIBXSMM_X86_INSTR_CMOVNAD, 0 );
  test_alu_reg( "alu_reg_CMOVNAQ", &mycode, LIBXSMM_X86_INSTR_CMOVNAQ, 0 );
  test_alu_reg( "alu_reg_CMOVNAEW", &mycode, LIBXSMM_X86_INSTR_CMOVNAEW, 0 );
  test_alu_reg( "alu_reg_CMOVNAED", &mycode, LIBXSMM_X86_INSTR_CMOVNAED, 0 );
  test_alu_reg( "alu_reg_CMOVNAEQ", &mycode, LIBXSMM_X86_INSTR_CMOVNAEQ, 0 );
  test_alu_reg( "alu_reg_CMOVNBW", &mycode, LIBXSMM_X86_INSTR_CMOVNBW, 0 );
  test_alu_reg( "alu_reg_CMOVNBD", &mycode, LIBXSMM_X86_INSTR_CMOVNBD, 0 );
  test_alu_reg( "alu_reg_CMOVNBQ", &mycode, LIBXSMM_X86_INSTR_CMOVNBQ, 0 );
  test_alu_reg( "alu_reg_CMOVNBEW", &mycode, LIBXSMM_X86_INSTR_CMOVNBEW, 0 );
  test_alu_reg( "alu_reg_CMOVNBED", &mycode, LIBXSMM_X86_INSTR_CMOVNBED, 0 );
  test_alu_reg( "alu_reg_CMOVNBEQ", &mycode, LIBXSMM_X86_INSTR_CMOVNBEQ, 0 );
  test_alu_reg( "alu_reg_CMOVNCW", &mycode, LIBXSMM_X86_INSTR_CMOVNCW, 0 );
  test_alu_reg( "alu_reg_CMOVNCD", &mycode, LIBXSMM_X86_INSTR_CMOVNCD, 0 );
  test_alu_reg( "alu_reg_CMOVNCQ", &mycode, LIBXSMM_X86_INSTR_CMOVNCQ, 0 );
  test_alu_reg( "alu_reg_CMOVNEW", &mycode, LIBXSMM_X86_INSTR_CMOVNEW, 0 );
  test_alu_reg( "alu_reg_CMOVNED", &mycode, LIBXSMM_X86_INSTR_CMOVNED, 0 );
  test_alu_reg( "alu_reg_CMOVNEQ", &mycode, LIBXSMM_X86_INSTR_CMOVNEQ, 0 );
  test_alu_reg( "alu_reg_CMOVNGW", &mycode, LIBXSMM_X86_INSTR_CMOVNGW, 0 );
  test_alu_reg( "alu_reg_CMOVNGD", &mycode, LIBXSMM_X86_INSTR_CMOVNGD, 0 );
  test_alu_reg( "alu_reg_CMOVNGQ", &mycode, LIBXSMM_X86_INSTR_CMOVNGQ, 0 );
  test_alu_reg( "alu_reg_CMOVNGEW", &mycode, LIBXSMM_X86_INSTR_CMOVNGEW, 0 );
  test_alu_reg( "alu_reg_CMOVNGED", &mycode, LIBXSMM_X86_INSTR_CMOVNGED, 0 );
  test_alu_reg( "alu_reg_CMOVNGEQ", &mycode, LIBXSMM_X86_INSTR_CMOVNGEQ, 0 );
  test_alu_reg( "alu_reg_CMOVNLW", &mycode, LIBXSMM_X86_INSTR_CMOVNLW, 0 );
  test_alu_reg( "alu_reg_CMOVNLD", &mycode, LIBXSMM_X86_INSTR_CMOVNLD, 0 );
  test_alu_reg( "alu_reg_CMOVNLQ", &mycode, LIBXSMM_X86_INSTR_CMOVNLQ, 0 );
  test_alu_reg( "alu_reg_CMOVNLEW", &mycode, LIBXSMM_X86_INSTR_CMOVNLEW, 0 );
  test_alu_reg( "alu_reg_CMOVNLED", &mycode, LIBXSMM_X86_INSTR_CMOVNLED, 0 );
  test_alu_reg( "alu_reg_CMOVNLEQ", &mycode, LIBXSMM_X86_INSTR_CMOVNLEQ, 0 );
  test_alu_reg( "alu_reg_CMOVNOW", &mycode, LIBXSMM_X86_INSTR_CMOVNOW, 0 );
  test_alu_reg( "alu_reg_CMOVNOD", &mycode, LIBXSMM_X86_INSTR_CMOVNOD, 0 );
  test_alu_reg( "alu_reg_CMOVNOQ", &mycode, LIBXSMM_X86_INSTR_CMOVNOQ, 0 );
  test_alu_reg( "alu_reg_CMOVNPW", &mycode, LIBXSMM_X86_INSTR_CMOVNPW, 0 );
  test_alu_reg( "alu_reg_CMOVNPD", &mycode, LIBXSMM_X86_INSTR_CMOVNPD, 0 );
  test_alu_reg( "alu_reg_CMOVNPQ", &mycode, LIBXSMM_X86_INSTR_CMOVNPQ, 0 );
  test_alu_reg( "alu_reg_CMOVNSW", &mycode, LIBXSMM_X86_INSTR_CMOVNSW, 0 );
  test_alu_reg( "alu_reg_CMOVNSD", &mycode, LIBXSMM_X86_INSTR_CMOVNSD, 0 );
  test_alu_reg( "alu_reg_CMOVNSQ", &mycode, LIBXSMM_X86_INSTR_CMOVNSQ, 0 );
  test_alu_reg( "alu_reg_CMOVNZW", &mycode, LIBXSMM_X86_INSTR_CMOVNZW, 0 );
  test_alu_reg( "alu_reg_CMOVNZD", &mycode, LIBXSMM_X86_INSTR_CMOVNZD, 0 );
  test_alu_reg( "alu_reg_CMOVNZQ", &mycode, LIBXSMM_X86_INSTR_CMOVNZQ, 0 );
  test_alu_reg( "alu_reg_CMOVOW", &mycode, LIBXSMM_X86_INSTR_CMOVOW, 0 );
  test_alu_reg( "alu_reg_CMOVOD", &mycode, LIBXSMM_X86_INSTR_CMOVOD, 0 );
  test_alu_reg( "alu_reg_CMOVOQ", &mycode, LIBXSMM_X86_INSTR_CMOVOQ, 0 );
  test_alu_reg( "alu_reg_CMOVPW", &mycode, LIBXSMM_X86_INSTR_CMOVPW, 0 );
  test_alu_reg( "alu_reg_CMOVPD", &mycode, LIBXSMM_X86_INSTR_CMOVPD, 0 );
  test_alu_reg( "alu_reg_CMOVPQ", &mycode, LIBXSMM_X86_INSTR_CMOVPQ, 0 );
  test_alu_reg( "alu_reg_CMOVPEW", &mycode, LIBXSMM_X86_INSTR_CMOVPEW, 0 );
  test_alu_reg( "alu_reg_CMOVPED", &mycode, LIBXSMM_X86_INSTR_CMOVPED, 0 );
  test_alu_reg( "alu_reg_CMOVPEQ", &mycode, LIBXSMM_X86_INSTR_CMOVPEQ, 0 );
  test_alu_reg( "alu_reg_CMOVPOW", &mycode, LIBXSMM_X86_INSTR_CMOVPOW, 0 );
  test_alu_reg( "alu_reg_CMOVPOD", &mycode, LIBXSMM_X86_INSTR_CMOVPOD, 0 );
  test_alu_reg( "alu_reg_CMOVPOQ", &mycode, LIBXSMM_X86_INSTR_CMOVPOQ, 0 );
  test_alu_reg( "alu_reg_CMOVSW", &mycode, LIBXSMM_X86_INSTR_CMOVSW, 0 );
  test_alu_reg( "alu_reg_CMOVSD", &mycode, LIBXSMM_X86_INSTR_CMOVSD, 0 );
  test_alu_reg( "alu_reg_CMOVSQ", &mycode, LIBXSMM_X86_INSTR_CMOVSQ, 0 );
  test_alu_reg( "alu_reg_CMOVZW", &mycode, LIBXSMM_X86_INSTR_CMOVZW, 0 );
  test_alu_reg( "alu_reg_CMOVZD", &mycode, LIBXSMM_X86_INSTR_CMOVZD, 0 );
  test_alu_reg( "alu_reg_CMOVZQ", &mycode, LIBXSMM_X86_INSTR_CMOVZQ, 0 );
  test_alu_reg( "alu_reg_CMPQ", &mycode, LIBXSMM_X86_INSTR_CMPQ, 0 );
  test_alu_reg( "alu_reg_CMPB_RM_R", &mycode, LIBXSMM_X86_INSTR_CMPB_RM_R, 0 );
  test_alu_reg( "alu_reg_CMPW_RM_R", &mycode, LIBXSMM_X86_INSTR_CMPW_RM_R, 0 );
  test_alu_reg( "alu_reg_CMPD_RM_R", &mycode, LIBXSMM_X86_INSTR_CMPD_RM_R, 0 );
  test_alu_reg( "alu_reg_CMPQ_RM_R", &mycode, LIBXSMM_X86_INSTR_CMPQ_RM_R, 0 );
  test_alu_reg( "alu_reg_CMPB_R_RM", &mycode, LIBXSMM_X86_INSTR_CMPB_R_RM, 0 );
  test_alu_reg( "alu_reg_CMPW_R_RM", &mycode, LIBXSMM_X86_INSTR_CMPW_R_RM, 0 );
  test_alu_reg( "alu_reg_CMPD_R_RM", &mycode, LIBXSMM_X86_INSTR_CMPD_R_RM, 0 );
  test_alu_reg( "alu_reg_CMPQ_R_RM", &mycode, LIBXSMM_X86_INSTR_CMPQ_R_RM, 0 );
  test_alu_reg( "alu_reg_IDIVW", &mycode, LIBXSMM_X86_INSTR_IDIVW, 1 );
  test_alu_reg( "alu_reg_IDIVD", &mycode, LIBXSMM_X86_INSTR_IDIVD, 1 );
  test_alu_reg( "alu_reg_IDIVQ", &mycode, LIBXSMM_X86_INSTR_IDIVQ, 1 );
  test_alu_reg( "alu_reg_IMULW", &mycode, LIBXSMM_X86_INSTR_IMULW, 0 );
  test_alu_reg( "alu_reg_IMULD", &mycode, LIBXSMM_X86_INSTR_IMULD, 0 );
  test_alu_reg( "alu_reg_IMULQ", &mycode, LIBXSMM_X86_INSTR_IMULQ, 0 );
  test_alu_reg( "alu_reg_LZCNTW", &mycode, LIBXSMM_X86_INSTR_LZCNTW, 0 );
  test_alu_reg( "alu_reg_LZCNTD", &mycode, LIBXSMM_X86_INSTR_LZCNTD, 0 );
  test_alu_reg( "alu_reg_LZCNTQ", &mycode, LIBXSMM_X86_INSTR_LZCNTQ, 0 );
  test_alu_reg( "alu_reg_MOVQ", &mycode, LIBXSMM_X86_INSTR_MOVQ, 0 );
  test_alu_reg( "alu_reg_MOVB_LD", &mycode, LIBXSMM_X86_INSTR_MOVB_LD, 0 );
  test_alu_reg( "alu_reg_MOVB_ST", &mycode, LIBXSMM_X86_INSTR_MOVB_ST, 0 );
  test_alu_reg( "alu_reg_MOVW_LD", &mycode, LIBXSMM_X86_INSTR_MOVW_LD, 0 );
  test_alu_reg( "alu_reg_MOVW_ST", &mycode, LIBXSMM_X86_INSTR_MOVW_ST, 0 );
  test_alu_reg( "alu_reg_MOVD_LD", &mycode, LIBXSMM_X86_INSTR_MOVD_LD, 0 );
  test_alu_reg( "alu_reg_MOVD_ST", &mycode, LIBXSMM_X86_INSTR_MOVD_ST, 0 );
  test_alu_reg( "alu_reg_MOVQ_LD", &mycode, LIBXSMM_X86_INSTR_MOVQ_LD, 0 );
  test_alu_reg( "alu_reg_MOVQ_ST", &mycode, LIBXSMM_X86_INSTR_MOVQ_ST, 0 );
  test_alu_reg( "alu_reg_NEGB", &mycode, LIBXSMM_X86_INSTR_NEGB, 1 );
  test_alu_reg( "alu_reg_NEGW", &mycode, LIBXSMM_X86_INSTR_NEGW, 1 );
  test_alu_reg( "alu_reg_NEGD", &mycode, LIBXSMM_X86_INSTR_NEGD, 1 );
  test_alu_reg( "alu_reg_NEGQ", &mycode, LIBXSMM_X86_INSTR_NEGQ, 1 );
  test_alu_reg( "alu_reg_NOTB", &mycode, LIBXSMM_X86_INSTR_NOTB, 1 );
  test_alu_reg( "alu_reg_NOTW", &mycode, LIBXSMM_X86_INSTR_NOTW, 1 );
  test_alu_reg( "alu_reg_NOTD", &mycode, LIBXSMM_X86_INSTR_NOTD, 1 );
  test_alu_reg( "alu_reg_NOTQ", &mycode, LIBXSMM_X86_INSTR_NOTQ, 1 );
  test_alu_reg( "alu_reg_ORB_RM_R", &mycode, LIBXSMM_X86_INSTR_ORB_RM_R, 0 );
  test_alu_reg( "alu_reg_ORW_RM_R", &mycode, LIBXSMM_X86_INSTR_ORW_RM_R, 0 );
  test_alu_reg( "alu_reg_ORD_RM_R", &mycode, LIBXSMM_X86_INSTR_ORD_RM_R, 0 );
  test_alu_reg( "alu_reg_ORQ_RM_R", &mycode, LIBXSMM_X86_INSTR_ORQ_RM_R, 0 );
  test_alu_reg( "alu_reg_ORB_R_RM", &mycode, LIBXSMM_X86_INSTR_ORB_R_RM, 0 );
  test_alu_reg( "alu_reg_ORW_R_RM", &mycode, LIBXSMM_X86_INSTR_ORW_R_RM, 0 );
  test_alu_reg( "alu_reg_ORD_R_RM", &mycode, LIBXSMM_X86_INSTR_ORD_R_RM, 0 );
  test_alu_reg( "alu_reg_ORQ_R_RM", &mycode, LIBXSMM_X86_INSTR_ORQ_R_RM, 0 );
  test_alu_reg( "alu_reg_POPW", &mycode, LIBXSMM_X86_INSTR_POPW, 1 );
  test_alu_reg( "alu_reg_POPQ", &mycode, LIBXSMM_X86_INSTR_POPQ, 1 );
  test_alu_reg( "alu_reg_POPW_RM", &mycode, LIBXSMM_X86_INSTR_POPW_RM, 1 );
  test_alu_reg( "alu_reg_POPQ_RM", &mycode, LIBXSMM_X86_INSTR_POPQ_RM, 1 );
  test_alu_reg( "alu_reg_POPCNT", &mycode, LIBXSMM_X86_INSTR_POPCNT, 0 );
  test_alu_reg( "alu_reg_POPCNTW", &mycode, LIBXSMM_X86_INSTR_POPCNTW, 0 );
  test_alu_reg( "alu_reg_POPCNTD", &mycode, LIBXSMM_X86_INSTR_POPCNTD, 0 );
  test_alu_reg( "alu_reg_POPCNTQ", &mycode, LIBXSMM_X86_INSTR_POPCNTQ, 0 );
  test_alu_reg( "alu_reg_PUSHW", &mycode, LIBXSMM_X86_INSTR_PUSHW, 1 );
  test_alu_reg( "alu_reg_PUSHQ", &mycode, LIBXSMM_X86_INSTR_PUSHQ, 1 );
  test_alu_reg( "alu_reg_PUSHW_RM", &mycode, LIBXSMM_X86_INSTR_PUSHW_RM, 1 );
  test_alu_reg( "alu_reg_PUSHQ_RM", &mycode, LIBXSMM_X86_INSTR_PUSHQ_RM, 1 );
  test_alu_reg( "alu_reg_SUBQ", &mycode, LIBXSMM_X86_INSTR_SUBQ, 0 );
  test_alu_reg( "alu_reg_SUBB_RM_R", &mycode, LIBXSMM_X86_INSTR_SUBB_RM_R, 0 );
  test_alu_reg( "alu_reg_SUBW_RM_R", &mycode, LIBXSMM_X86_INSTR_SUBW_RM_R, 0 );
  test_alu_reg( "alu_reg_SUBD_RM_R", &mycode, LIBXSMM_X86_INSTR_SUBD_RM_R, 0 );
  test_alu_reg( "alu_reg_SUBQ_RM_R", &mycode, LIBXSMM_X86_INSTR_SUBQ_RM_R, 0 );
  test_alu_reg( "alu_reg_SUBB_R_RM", &mycode, LIBXSMM_X86_INSTR_SUBB_R_RM, 0 );
  test_alu_reg( "alu_reg_SUBW_R_RM", &mycode, LIBXSMM_X86_INSTR_SUBW_R_RM, 0 );
  test_alu_reg( "alu_reg_SUBD_R_RM", &mycode, LIBXSMM_X86_INSTR_SUBD_R_RM, 0 );
  test_alu_reg( "alu_reg_SUBQ_R_RM", &mycode, LIBXSMM_X86_INSTR_SUBQ_R_RM, 0 );
  test_alu_reg( "alu_reg_TZCNT", &mycode, LIBXSMM_X86_INSTR_TZCNT, 0 );
  test_alu_reg( "alu_reg_TZCNTW", &mycode, LIBXSMM_X86_INSTR_TZCNTW, 0 );
  test_alu_reg( "alu_reg_TZCNTD", &mycode, LIBXSMM_X86_INSTR_TZCNTD, 0 );
  test_alu_reg( "alu_reg_TZCNTQ", &mycode, LIBXSMM_X86_INSTR_TZCNTQ, 0 );
  test_alu_reg( "alu_reg_XORB_RM_R", &mycode, LIBXSMM_X86_INSTR_XORB_RM_R, 0 );
  test_alu_reg( "alu_reg_XORW_RM_R", &mycode, LIBXSMM_X86_INSTR_XORW_RM_R, 0 );
  test_alu_reg( "alu_reg_XORD_RM_R", &mycode, LIBXSMM_X86_INSTR_XORD_RM_R, 0 );
  test_alu_reg( "alu_reg_XORQ_RM_R", &mycode, LIBXSMM_X86_INSTR_XORQ_RM_R, 0 );
  test_alu_reg( "alu_reg_XORB_R_RM", &mycode, LIBXSMM_X86_INSTR_XORB_R_RM, 0 );
  test_alu_reg( "alu_reg_XORW_R_RM", &mycode, LIBXSMM_X86_INSTR_XORW_R_RM, 0 );
  test_alu_reg( "alu_reg_XORD_R_RM", &mycode, LIBXSMM_X86_INSTR_XORD_R_RM, 0 );
  test_alu_reg( "alu_reg_XORQ_R_RM", &mycode, LIBXSMM_X86_INSTR_XORQ_R_RM, 0 );

  /* test alu mem */
  test_alu_mem( "alu_mov_MOVB_LD", &mycode, LIBXSMM_X86_INSTR_MOVB, 1 );
  test_alu_mem( "alu_mov_MOVB_ST", &mycode, LIBXSMM_X86_INSTR_MOVB, 2 );
  test_alu_mem( "alu_mov_MOVW_LD", &mycode, LIBXSMM_X86_INSTR_MOVW, 1 );
  test_alu_mem( "alu_mov_MOVW_ST", &mycode, LIBXSMM_X86_INSTR_MOVW, 2 );
  test_alu_mem( "alu_mov_MOVD_LD", &mycode, LIBXSMM_X86_INSTR_MOVD, 1 );
  test_alu_mem( "alu_mov_MOVD_ST", &mycode, LIBXSMM_X86_INSTR_MOVD, 2 );
  test_alu_mem( "alu_mov_MOVQ_LD", &mycode, LIBXSMM_X86_INSTR_MOVQ, 1 );
  test_alu_mem( "alu_mov_MOVQ_ST", &mycode, LIBXSMM_X86_INSTR_MOVQ, 2 );
  test_alu_mem( "alu_mov_LEAW", &mycode, LIBXSMM_X86_INSTR_LEAW, 1 );
  test_alu_mem( "alu_mov_LEAD", &mycode, LIBXSMM_X86_INSTR_LEAD, 1 );
  test_alu_mem( "alu_mov_LEAQ", &mycode, LIBXSMM_X86_INSTR_LEAQ, 1 );

  /* test alu imm */
  test_alu_imm( "alu_imm_ADDQ",          &mycode, LIBXSMM_X86_INSTR_ADDQ );
  test_alu_imm( "alu_imm_ADDB_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_ADDB_RM_IMM8 );
  test_alu_imm( "alu_imm_ADDW_RM_IMM16", &mycode, LIBXSMM_X86_INSTR_ADDW_RM_IMM16 );
  test_alu_imm( "alu_imm_ADDD_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_ADDD_RM_IMM32 );
  test_alu_imm( "alu_imm_ADDQ_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_ADDQ_RM_IMM32 );
  test_alu_imm( "alu_imm_ANDQ", &mycode, LIBXSMM_X86_INSTR_ANDQ );
  test_alu_imm( "alu_imm_ANDB_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_ANDB_RM_IMM8 );
  test_alu_imm( "alu_imm_ANDW_RM_IMM16", &mycode, LIBXSMM_X86_INSTR_ANDW_RM_IMM16 );
  test_alu_imm( "alu_imm_ANDD_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_ANDD_RM_IMM32 );
  test_alu_imm( "alu_imm_ANDQ_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_ANDQ_RM_IMM32 );
  test_alu_imm( "alu_imm_CMPQ", &mycode, LIBXSMM_X86_INSTR_CMPQ );
  test_alu_imm( "alu_imm_CMPB_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_CMPB_RM_IMM8 );
  test_alu_imm( "alu_imm_CMPW_RM_IMM16", &mycode, LIBXSMM_X86_INSTR_CMPW_RM_IMM16 );
  test_alu_imm( "alu_imm_CMPD_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_CMPD_RM_IMM32 );
  test_alu_imm( "alu_imm_CMPQ_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_CMPQ_RM_IMM32 );
  test_alu_imm( "alu_imm_IMUL", &mycode, LIBXSMM_X86_INSTR_IMUL );
  test_alu_imm( "alu_imm_IMULW_IMM16",   &mycode, LIBXSMM_X86_INSTR_IMULW_IMM16 );
  test_alu_imm( "alu_imm_IMULD_IMM32",   &mycode, LIBXSMM_X86_INSTR_IMULD_IMM32 );
  test_alu_imm( "alu_imm_IMULQ_IMM32",   &mycode, LIBXSMM_X86_INSTR_IMULQ_IMM32 );
  test_alu_imm( "alu_imm_MOVQ", &mycode, LIBXSMM_X86_INSTR_MOVQ );
  test_alu_imm( "alu_imm_MOVB_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_MOVB_RM_IMM8 );
  test_alu_imm( "alu_imm_MOVW_RM_IMM16", &mycode, LIBXSMM_X86_INSTR_MOVW_RM_IMM16 );
  test_alu_imm( "alu_imm_MOVD_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_MOVD_RM_IMM32 );
  test_alu_imm( "alu_imm_MOVQ_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_MOVQ_RM_IMM32 );
  test_alu_imm( "alu_imm_ORB_RM_IMM8",   &mycode, LIBXSMM_X86_INSTR_ORB_RM_IMM8 );
  test_alu_imm( "alu_imm_ORW_RM_IMM16",  &mycode, LIBXSMM_X86_INSTR_ORW_RM_IMM16 );
  test_alu_imm( "alu_imm_ORD_RM_IMM32",  &mycode, LIBXSMM_X86_INSTR_ORD_RM_IMM32 );
  test_alu_imm( "alu_imm_ORQ_RM_IMM32",  &mycode, LIBXSMM_X86_INSTR_ORQ_RM_IMM32 );
  test_alu_imm( "alu_imm_SALQ", &mycode, LIBXSMM_X86_INSTR_SALQ );
  test_alu_imm( "alu_imm_SALB_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SALB_RM_IMM8 );
  test_alu_imm( "alu_imm_SALW_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SALW_RM_IMM8 );
  test_alu_imm( "alu_imm_SALD_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SALD_RM_IMM8 );
  test_alu_imm( "alu_imm_SALQ_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SALQ_RM_IMM8 );
  test_alu_imm( "alu_imm_SARQ", &mycode, LIBXSMM_X86_INSTR_SARQ );
  test_alu_imm( "alu_imm_SARB_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SARB_RM_IMM8 );
  test_alu_imm( "alu_imm_SARW_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SARW_RM_IMM8 );
  test_alu_imm( "alu_imm_SARD_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SARD_RM_IMM8 );
  test_alu_imm( "alu_imm_SARQ_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SARQ_RM_IMM8 );
  test_alu_imm( "alu_imm_SHLQ", &mycode, LIBXSMM_X86_INSTR_SHLQ );
  test_alu_imm( "alu_imm_SHLB_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SHLB_RM_IMM8 );
  test_alu_imm( "alu_imm_SHLW_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SHLW_RM_IMM8 );
  test_alu_imm( "alu_imm_SHLD_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SHLD_RM_IMM8 );
  test_alu_imm( "alu_imm_SHLQ_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SHLQ_RM_IMM8 );
  test_alu_imm( "alu_imm_SHRQ", &mycode, LIBXSMM_X86_INSTR_SHRQ );
  test_alu_imm( "alu_imm_SHRB_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SHRB_RM_IMM8 );
  test_alu_imm( "alu_imm_SHRW_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SHRW_RM_IMM8 );
  test_alu_imm( "alu_imm_SHRD_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SHRD_RM_IMM8 );
  test_alu_imm( "alu_imm_SHRQ_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SHRQ_RM_IMM8 );
  test_alu_imm( "alu_imm_SUBQ", &mycode, LIBXSMM_X86_INSTR_SUBQ );
  test_alu_imm( "alu_imm_SUBB_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_SUBB_RM_IMM8 );
  test_alu_imm( "alu_imm_SUBW_RM_IMM16", &mycode, LIBXSMM_X86_INSTR_SUBW_RM_IMM16 );
  test_alu_imm( "alu_imm_SUBD_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_SUBD_RM_IMM32 );
  test_alu_imm( "alu_imm_SUBQ_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_SUBQ_RM_IMM32 );
  test_alu_imm( "alu_imm_XORB_RM_IMM8",  &mycode, LIBXSMM_X86_INSTR_XORB_RM_IMM8 );
  test_alu_imm( "alu_imm_XORW_RM_IMM16", &mycode, LIBXSMM_X86_INSTR_XORW_RM_IMM16 );
  test_alu_imm( "alu_imm_XORD_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_XORD_RM_IMM32 );
  test_alu_imm( "alu_imm_XORQ_RM_IMM32", &mycode, LIBXSMM_X86_INSTR_XORQ_RM_IMM32 );

  test_alu_imm_i64( "alu_imm_MOVQ_IMM64", &mycode, LIBXSMM_X86_INSTR_MOVQ );
  test_alu_imm_i64( "alu_imm_MOVB_R_IMM8",  &mycode, LIBXSMM_X86_INSTR_MOVB_R_IMM8 );
  test_alu_imm_i64( "alu_imm_MOVW_R_IMM16", &mycode, LIBXSMM_X86_INSTR_MOVW_R_IMM16 );
  test_alu_imm_i64( "alu_imm_MOVD_R_IMM32", &mycode, LIBXSMM_X86_INSTR_MOVD_R_IMM32 );
  test_alu_imm_i64( "alu_imm_MOVQ_R_IMM64", &mycode, LIBXSMM_X86_INSTR_MOVQ_R_IMM64 );

  test_alu_stack( "alu_stack_PUSHQ", &mycode, 0 );
  test_alu_stack( "alu_stack_POPQ", &mycode, 1 );

  free( codebuffer );

  return 0;
}
