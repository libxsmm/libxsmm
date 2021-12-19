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

void test_evex_compute_mem_2reg_general( char* test_name, libxsmm_generated_code* mycode, unsigned int instr, unsigned int reg_args, unsigned short imm8, unsigned int max_dst, unsigned no_mask ) {
  unsigned int i;
  unsigned int m;
  unsigned int b;
  unsigned int scale = 2;
  int displ[3] = {0, 128, 2097152};
  unsigned int d;
  unsigned int z;
  unsigned int init_dst = ( max_dst == 8 ) ? 1 : 0;
  unsigned int mloop = ( no_mask == 0 ) ? 8 : 1;

  reset_code_buffer( mycode, test_name );

  for (b = 0; b < 16; ++b ) {
    for (z = init_dst; z < max_dst; ++z ) {
      for ( m = 0; m < mloop; ++m ) {
        for (d = 0; d < 3; ++d ) {
          if ( reg_args == 0 ) {
            libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, LIBXSMM_X86_GP_REG_UNDEF, 0, displ[d], 0, LIBXSMM_X86_VEC_REG_UNDEF, LIBXSMM_X86_VEC_REG_UNDEF, m, 0, imm8 );
          } else if ( reg_args == 1 ) {
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
      for (z = init_dst; z < max_dst; ++z ) {
        for ( m = 0; m < mloop; ++m ) {
          for (d = 0; d < 3; ++d ) {
            if ( reg_args == 0 ) {
              libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8 ( mycode, instr, 'z', b, i, scale, displ[d], 0, LIBXSMM_X86_VEC_REG_UNDEF, LIBXSMM_X86_VEC_REG_UNDEF, m, 0, imm8 );
            } else if ( reg_args == 1 ) {
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

void test_alu_reg( char* test_name, libxsmm_generated_code* mycode, unsigned int instr ) {
  unsigned int t;

  reset_code_buffer( mycode, test_name );

  for (t = 0; t < 16; ++t ) {
    libxsmm_x86_instruction_alu_reg ( mycode, instr, t, 0 );
  }
  for (t = 0; t < 16; ++t ) {
    libxsmm_x86_instruction_alu_reg ( mycode, instr, 0, t );
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

int main( /*int argc, char* argv[]*/ ) {
  unsigned char* codebuffer = (unsigned char*)malloc( 8388608*sizeof(unsigned char) );
  libxsmm_generated_code mycode;

  /* init generated code object */
  mycode.generated_code = codebuffer;
  mycode.buffer_size = 8388608;
  mycode.arch = LIBXSMM_X86_AVX512_SPR;

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
  /* @TODO check these for stores */
  test_evex_load_store( "evex_mov_VPMOVDW", &mycode, LIBXSMM_X86_INSTR_VPMOVDW, 1 );
  test_evex_load_store( "evex_mov_VPMOVDB", &mycode, LIBXSMM_X86_INSTR_VPMOVDB, 1 );
  test_evex_load_store( "evex_mov_VPMOVSDB", &mycode, LIBXSMM_X86_INSTR_VPMOVSDB, 1 );
  test_evex_load_store( "evex_mov_VPMOVUSDB", &mycode, LIBXSMM_X86_INSTR_VPMOVUSDB, 1 );
  test_evex_load_store( "evex_mov_VPMOVSXWD", &mycode, LIBXSMM_X86_INSTR_VPMOVSXWD, 1 );
  test_evex_load_store( "evex_mov_VPMOVZXWD", &mycode, LIBXSMM_X86_INSTR_VPMOVZXWD, 1 );
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
  test_evex_compute_3reg_general( "evex_reg_VINSERTI32X4", &mycode, LIBXSMM_X86_INSTR_VINSERTI32X4, 1, 0x01, 32 );
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
  test_vex_compute_3reg_general( "vex_reg_VROUNDPD", &mycode, LIBXSMM_X86_INSTR_VROUNDPD, 1, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VROUNDSD", &mycode, LIBXSMM_X86_INSTR_VROUNDSD, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VROUNDPS", &mycode, LIBXSMM_X86_INSTR_VROUNDPS, 1, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VROUNDSS", &mycode, LIBXSMM_X86_INSTR_VROUNDSS, 0, 0x01, 0 );
  test_vex_compute_3reg_general( "vex_reg_VRCPPS", &mycode, LIBXSMM_X86_INSTR_VRCPPS, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VRCPSS", &mycode, LIBXSMM_X86_INSTR_VRCPSS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VRSQRTPS", &mycode, LIBXSMM_X86_INSTR_VRSQRTPS, 1, LIBXSMM_X86_IMM_UNDEF, 0 );
  test_vex_compute_3reg_general( "vex_reg_VRSQRTSS", &mycode, LIBXSMM_X86_INSTR_VRSQRTSS, 0, LIBXSMM_X86_IMM_UNDEF, 0 );
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
  test_vex_compute_3reg_general( "vex_reg_VPCMPESTRI", &mycode, LIBXSMM_X86_INSTR_VPCMPESTRI, 1, 0x01, 1 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPESTRM", &mycode, LIBXSMM_X86_INSTR_VPCMPESTRM, 1, 0x01, 1 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPISTRI", &mycode, LIBXSMM_X86_INSTR_VPCMPISTRI, 1, 0x01, 1 );
  test_vex_compute_3reg_general( "vex_reg_VPCMPISTRM", &mycode, LIBXSMM_X86_INSTR_VPCMPISTRM, 1, 0x01, 1 );
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
  test_evex_compute_3reg_general( "evex_reg_VPMOVSXBD", &mycode, LIBXSMM_X86_INSTR_VPMOVSXBD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVZXBD", &mycode, LIBXSMM_X86_INSTR_VPMOVZXBD, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVUSWB", &mycode, LIBXSMM_X86_INSTR_VPMOVUSWB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVSWB", &mycode, LIBXSMM_X86_INSTR_VPMOVSWB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPMOVWB", &mycode, LIBXSMM_X86_INSTR_VPMOVWB, 1, LIBXSMM_X86_IMM_UNDEF, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSLLD_I", &mycode, LIBXSMM_X86_INSTR_VPSLLD_I, 1, 0x01, 32 );
  test_evex_compute_3reg_general( "evex_reg_VPSRAD_I", &mycode, LIBXSMM_X86_INSTR_VPSRAD_I, 1, 0x01, 32 );
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
  mycode.arch = LIBXSMM_X86_AVX2;
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
  mycode.arch = LIBXSMM_X86_AVX512_SPR;

  /* testing compute mem-reg instructions */
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFPS", &mycode, LIBXSMM_X86_INSTR_VSHUFPS, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFPD", &mycode, LIBXSMM_X86_INSTR_VSHUFPD, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSHUFB", &mycode, LIBXSMM_X86_INSTR_VPSHUFB, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSHUFD", &mycode, LIBXSMM_X86_INSTR_VPSHUFD, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSHUFHW", &mycode, LIBXSMM_X86_INSTR_VPSHUFHW, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSHUFLW", &mycode, LIBXSMM_X86_INSTR_VPSHUFLW, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFF32X4", &mycode, LIBXSMM_X86_INSTR_VSHUFF32X4, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFF64X2", &mycode, LIBXSMM_X86_INSTR_VSHUFF64X2, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFI32X4", &mycode, LIBXSMM_X86_INSTR_VSHUFI32X4, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSHUFI64X2", &mycode, LIBXSMM_X86_INSTR_VSHUFI64X2, 2, 0x01, 32, 0 );
  test_vex_compute_mem_2reg_general( "evex_mem_VEXTRACTF128", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF128, 1, 0x01, 0 );
  test_vex_compute_mem_2reg_general( "evex_mem_VEXTRACTI128", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI128, 1, 0x01, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTF32X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF32X4, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTF64X2", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF64X2, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTF32X8", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF32X8, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTF64X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTF64X4, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTI32X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI32X4, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTI64X2", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI64X2, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTI32X8", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI32X8, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXTRACTI64X4", &mycode, LIBXSMM_X86_INSTR_VEXTRACTI64X4, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VINSERTI32X4", &mycode, LIBXSMM_X86_INSTR_VINSERTI32X4, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VBLENDMPS", &mycode, LIBXSMM_X86_INSTR_VBLENDMPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VBLENDMPD", &mycode, LIBXSMM_X86_INSTR_VBLENDMPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPBLENDMB", &mycode, LIBXSMM_X86_INSTR_VPBLENDMB, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPBLENDMW", &mycode, LIBXSMM_X86_INSTR_VPBLENDMW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPBLENDMD", &mycode, LIBXSMM_X86_INSTR_VPBLENDMD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPBLENDMQ", &mycode, LIBXSMM_X86_INSTR_VPBLENDMQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXPANDPD", &mycode, LIBXSMM_X86_INSTR_VEXPANDPD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VEXPANDPS", &mycode, LIBXSMM_X86_INSTR_VEXPANDPS, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPEXPANDQ", &mycode, LIBXSMM_X86_INSTR_VPEXPANDQ, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPEXPANDD", &mycode, LIBXSMM_X86_INSTR_VPEXPANDD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPEXPANDW", &mycode, LIBXSMM_X86_INSTR_VPEXPANDW, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPEXPANDB", &mycode, LIBXSMM_X86_INSTR_VPEXPANDB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VUNPCKLPD", &mycode, LIBXSMM_X86_INSTR_VUNPCKLPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VUNPCKLPS", &mycode, LIBXSMM_X86_INSTR_VUNPCKLPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VUNPCKHPD", &mycode, LIBXSMM_X86_INSTR_VUNPCKHPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VUNPCKHPS", &mycode, LIBXSMM_X86_INSTR_VUNPCKHPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKLWD", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLWD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKHWD", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHWD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKLDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKHDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKLQDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKLQDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPUNPCKHQDQ", &mycode, LIBXSMM_X86_INSTR_VPUNPCKHQDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_vex_compute_mem_2reg_general( "evex_mem_VPERM2F128", &mycode, LIBXSMM_X86_INSTR_VPERM2F128, 2, 0x01, 0 );
  test_vex_compute_mem_2reg_general( "evex_mem_VPERM2I128", &mycode, LIBXSMM_X86_INSTR_VPERM2I128, 2, 0x01, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMW", &mycode, LIBXSMM_X86_INSTR_VPERMW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMD", &mycode, LIBXSMM_X86_INSTR_VPERMD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMQ_I", &mycode, LIBXSMM_X86_INSTR_VPERMQ_I, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMT2B", &mycode, LIBXSMM_X86_INSTR_VPERMT2B, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMT2W", &mycode, LIBXSMM_X86_INSTR_VPERMT2W, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMT2D", &mycode, LIBXSMM_X86_INSTR_VPERMT2D, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPERMT2Q", &mycode, LIBXSMM_X86_INSTR_VPERMT2Q, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD132PS", &mycode, LIBXSMM_X86_INSTR_VFMADD132PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD132PD", &mycode, LIBXSMM_X86_INSTR_VFMADD132PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD213PS", &mycode, LIBXSMM_X86_INSTR_VFMADD213PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD213PD", &mycode, LIBXSMM_X86_INSTR_VFMADD213PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD231PS", &mycode, LIBXSMM_X86_INSTR_VFMADD231PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD231PD", &mycode, LIBXSMM_X86_INSTR_VFMADD231PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB132PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB132PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB132PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB132PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB213PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB213PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB213PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB213PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB231PS", &mycode, LIBXSMM_X86_INSTR_VFMSUB231PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB231PD", &mycode, LIBXSMM_X86_INSTR_VFMSUB231PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD132PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD132PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD132PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD132PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD213PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD213PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD213PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD213PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD231PS", &mycode, LIBXSMM_X86_INSTR_VFNMADD231PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD231PD", &mycode, LIBXSMM_X86_INSTR_VFNMADD231PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB132PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB132PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB213PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB213PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB231PS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB231PD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231PD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD132SD", &mycode, LIBXSMM_X86_INSTR_VFMADD132SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD213SD", &mycode, LIBXSMM_X86_INSTR_VFMADD213SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD231SD", &mycode, LIBXSMM_X86_INSTR_VFMADD231SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD132SS", &mycode, LIBXSMM_X86_INSTR_VFMADD132SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD213SS", &mycode, LIBXSMM_X86_INSTR_VFMADD213SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMADD231SS", &mycode, LIBXSMM_X86_INSTR_VFMADD231SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB132SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB132SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB213SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB213SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB231SD", &mycode, LIBXSMM_X86_INSTR_VFMSUB231SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB132SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB132SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB213SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB213SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFMSUB231SS", &mycode, LIBXSMM_X86_INSTR_VFMSUB231SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD132SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD132SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD213SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD213SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD231SD", &mycode, LIBXSMM_X86_INSTR_VFNMADD231SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD132SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD132SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD213SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD213SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMADD231SS", &mycode, LIBXSMM_X86_INSTR_VFNMADD231SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB132SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB213SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB231SD", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB132SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB132SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB213SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB213SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VFNMSUB231SS", &mycode, LIBXSMM_X86_INSTR_VFNMSUB231SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRANGEPS", &mycode, LIBXSMM_X86_INSTR_VRANGEPS, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRANGEPD", &mycode, LIBXSMM_X86_INSTR_VRANGEPD, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRANGESS", &mycode, LIBXSMM_X86_INSTR_VRANGESS, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRANGESD", &mycode, LIBXSMM_X86_INSTR_VRANGESD, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VREDUCEPS", &mycode, LIBXSMM_X86_INSTR_VREDUCEPS, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VREDUCEPD", &mycode, LIBXSMM_X86_INSTR_VREDUCEPD, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VREDUCESS", &mycode, LIBXSMM_X86_INSTR_VREDUCESS, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VREDUCESD", &mycode, LIBXSMM_X86_INSTR_VREDUCESD, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRCP14PS", &mycode, LIBXSMM_X86_INSTR_VRCP14PS, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRCP14PD", &mycode, LIBXSMM_X86_INSTR_VRCP14PD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRCP14SS", &mycode, LIBXSMM_X86_INSTR_VRCP14SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRCP14SD", &mycode, LIBXSMM_X86_INSTR_VRCP14SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRNDSCALEPS", &mycode, LIBXSMM_X86_INSTR_VRNDSCALEPS, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRNDSCALEPD", &mycode, LIBXSMM_X86_INSTR_VRNDSCALEPD, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRNDSCALESS", &mycode, LIBXSMM_X86_INSTR_VRNDSCALESS, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRNDSCALESD", &mycode, LIBXSMM_X86_INSTR_VRNDSCALESD, 2, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRSQRT14PS", &mycode, LIBXSMM_X86_INSTR_VRSQRT14PS, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRSQRT14PD", &mycode, LIBXSMM_X86_INSTR_VRSQRT14PD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRSQRT14SS", &mycode, LIBXSMM_X86_INSTR_VRSQRT14SS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VRSQRT14SD", &mycode, LIBXSMM_X86_INSTR_VRSQRT14SD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSCALEFPS", &mycode, LIBXSMM_X86_INSTR_VSCALEFPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSCALEFPD", &mycode, LIBXSMM_X86_INSTR_VSCALEFPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSCALEFSS", &mycode, LIBXSMM_X86_INSTR_VSCALEFSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSCALEFSD", &mycode, LIBXSMM_X86_INSTR_VSCALEFSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCMPPS", &mycode, LIBXSMM_X86_INSTR_VCMPPS, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCMPSS", &mycode, LIBXSMM_X86_INSTR_VCMPSS, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCMPPD", &mycode, LIBXSMM_X86_INSTR_VCMPPD, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCMPSD", &mycode, LIBXSMM_X86_INSTR_VCMPSD, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPB", &mycode, LIBXSMM_X86_INSTR_VPCMPB, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPUB", &mycode, LIBXSMM_X86_INSTR_VPCMPUB, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPW", &mycode, LIBXSMM_X86_INSTR_VPCMPW, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPUW", &mycode, LIBXSMM_X86_INSTR_VPCMPUW, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPD", &mycode, LIBXSMM_X86_INSTR_VPCMPD, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPUD", &mycode, LIBXSMM_X86_INSTR_VPCMPUD, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPQ", &mycode, LIBXSMM_X86_INSTR_VPCMPQ, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPUQ", &mycode, LIBXSMM_X86_INSTR_VPCMPUQ, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPEQB", &mycode, LIBXSMM_X86_INSTR_VPCMPEQB, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPEQW", &mycode, LIBXSMM_X86_INSTR_VPCMPEQW, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPEQD", &mycode, LIBXSMM_X86_INSTR_VPCMPEQD, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPEQQ", &mycode, LIBXSMM_X86_INSTR_VPCMPEQQ, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPGTB", &mycode, LIBXSMM_X86_INSTR_VPCMPGTB, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPGTW", &mycode, LIBXSMM_X86_INSTR_VPCMPGTW, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPGTD", &mycode, LIBXSMM_X86_INSTR_VPCMPGTD, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPCMPGTQ", &mycode, LIBXSMM_X86_INSTR_VPCMPGTQ, 2, 0x01, 8, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTPS2PD", &mycode, LIBXSMM_X86_INSTR_VCVTPS2PD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTPH2PS", &mycode, LIBXSMM_X86_INSTR_VCVTPH2PS, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTPS2PH", &mycode, LIBXSMM_X86_INSTR_VCVTPS2PH, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTDQ2PS", &mycode, LIBXSMM_X86_INSTR_VCVTDQ2PS, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTPS2DQ", &mycode, LIBXSMM_X86_INSTR_VCVTPS2DQ, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTPS2UDQ", &mycode, LIBXSMM_X86_INSTR_VCVTPS2UDQ, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVDW", &mycode, LIBXSMM_X86_INSTR_VPMOVDW, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVSXWD", &mycode, LIBXSMM_X86_INSTR_VPMOVSXWD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVDB", &mycode, LIBXSMM_X86_INSTR_VPMOVDB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVSDB", &mycode, LIBXSMM_X86_INSTR_VPMOVSDB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVUSDB", &mycode, LIBXSMM_X86_INSTR_VPMOVUSDB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVZXWD", &mycode, LIBXSMM_X86_INSTR_VPMOVZXWD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVSXBD", &mycode, LIBXSMM_X86_INSTR_VPMOVSXBD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVZXBD", &mycode, LIBXSMM_X86_INSTR_VPMOVZXBD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVUSWB", &mycode, LIBXSMM_X86_INSTR_VPMOVUSWB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVSWB", &mycode, LIBXSMM_X86_INSTR_VPMOVSWB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMOVWB", &mycode, LIBXSMM_X86_INSTR_VPMOVWB, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSLLD_I", &mycode, LIBXSMM_X86_INSTR_VPSLLD_I, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSRAD_I", &mycode, LIBXSMM_X86_INSTR_VPSRAD_I, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSRLD_I", &mycode, LIBXSMM_X86_INSTR_VPSRLD_I, 1, 0x01, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSRAVD", &mycode, LIBXSMM_X86_INSTR_VPSRAVD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VXORPD", &mycode, LIBXSMM_X86_INSTR_VXORPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VADDPD", &mycode, LIBXSMM_X86_INSTR_VADDPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMULPD", &mycode, LIBXSMM_X86_INSTR_VMULPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSUBPD", &mycode, LIBXSMM_X86_INSTR_VSUBPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VDIVPD", &mycode, LIBXSMM_X86_INSTR_VDIVPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMAXPD", &mycode, LIBXSMM_X86_INSTR_VMAXPD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VADDSD", &mycode, LIBXSMM_X86_INSTR_VADDSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMULSD", &mycode, LIBXSMM_X86_INSTR_VMULSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSUBSD", &mycode, LIBXSMM_X86_INSTR_VSUBSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VXORPS", &mycode, LIBXSMM_X86_INSTR_VXORPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VADDPS", &mycode, LIBXSMM_X86_INSTR_VADDPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMULPS", &mycode, LIBXSMM_X86_INSTR_VMULPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSUBPS", &mycode, LIBXSMM_X86_INSTR_VSUBPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VDIVPS", &mycode, LIBXSMM_X86_INSTR_VDIVPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMAXPS", &mycode, LIBXSMM_X86_INSTR_VMAXPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMULSS", &mycode, LIBXSMM_X86_INSTR_VMULSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VADDSS", &mycode, LIBXSMM_X86_INSTR_VADDSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VSUBSS", &mycode, LIBXSMM_X86_INSTR_VSUBSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPXORD", &mycode, LIBXSMM_X86_INSTR_VPXORD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPORD", &mycode, LIBXSMM_X86_INSTR_VPORD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPANDD", &mycode, LIBXSMM_X86_INSTR_VPANDD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPANDQ", &mycode, LIBXSMM_X86_INSTR_VPANDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDQ", &mycode, LIBXSMM_X86_INSTR_VPADDQ, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDB", &mycode, LIBXSMM_X86_INSTR_VPADDB, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDW", &mycode, LIBXSMM_X86_INSTR_VPADDW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDD", &mycode, LIBXSMM_X86_INSTR_VPADDD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMADDWD", &mycode, LIBXSMM_X86_INSTR_VPMADDWD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMADDUBSW", &mycode, LIBXSMM_X86_INSTR_VPMADDUBSW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDSW", &mycode, LIBXSMM_X86_INSTR_VPADDSW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPADDSB", &mycode, LIBXSMM_X86_INSTR_VPADDSB, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPSUBD", &mycode, LIBXSMM_X86_INSTR_VPSUBD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMAXSD", &mycode, LIBXSMM_X86_INSTR_VPMAXSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMAXSW", &mycode, LIBXSMM_X86_INSTR_VPMAXSW, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPMINSD", &mycode, LIBXSMM_X86_INSTR_VPMINSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_V4FMADDPS", &mycode, LIBXSMM_X86_INSTR_V4FMADDPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_V4FNMADDPS", &mycode, LIBXSMM_X86_INSTR_V4FNMADDPS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_V4FMADDSS", &mycode, LIBXSMM_X86_INSTR_V4FMADDSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_V4FNMADDSS", &mycode, LIBXSMM_X86_INSTR_V4FNMADDSS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VP4DPWSSDS", &mycode, LIBXSMM_X86_INSTR_VP4DPWSSDS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VP4DPWSSD", &mycode, LIBXSMM_X86_INSTR_VP4DPWSSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPDPBUSD", &mycode, LIBXSMM_X86_INSTR_VPDPBUSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPDPBUSDS", &mycode, LIBXSMM_X86_INSTR_VPDPBUSDS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPDPWSSD", &mycode, LIBXSMM_X86_INSTR_VPDPWSSD, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VPDPWSSDS", &mycode, LIBXSMM_X86_INSTR_VPDPWSSDS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VDPBF16PS", &mycode, LIBXSMM_X86_INSTR_VDPBF16PS, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTNEPS2BF16", &mycode, LIBXSMM_X86_INSTR_VCVTNEPS2BF16, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VCVTNE2PS2BF16", &mycode, LIBXSMM_X86_INSTR_VCVTNE2PS2BF16, 2, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMOVDQU64_LD", &mycode, LIBXSMM_X86_INSTR_VMOVDQU64_LD, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );
  test_evex_compute_mem_2reg_general( "evex_mem_VMOVDQU64_ST", &mycode, LIBXSMM_X86_INSTR_VMOVDQU64_ST, 1, LIBXSMM_X86_IMM_UNDEF, 32, 0 );

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
  test_prefetch( "pf_CLDEMOTE", &mycode, LIBXSMM_X86_INSTR_CLDEMOTE );
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

  /* AVX only tests */
  mycode.arch = LIBXSMM_X86_AVX2;

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

  /* SSE only test */
  test_rex_vload_vstore( "rex_mov_MOVAPD", &mycode, LIBXSMM_X86_INSTR_MOVAPD, 3 );
  test_rex_vload_vstore( "rex_mov_MOVUPD", &mycode, LIBXSMM_X86_INSTR_MOVUPD, 3 );
  test_rex_vload_vstore( "rex_mov_MOVAPS", &mycode, LIBXSMM_X86_INSTR_MOVAPS, 3 );
  test_rex_vload_vstore( "rex_mov_MOVUPS", &mycode, LIBXSMM_X86_INSTR_MOVUPS, 3 );
  test_rex_vload_vstore( "rex_mov_MOVSD", &mycode, LIBXSMM_X86_INSTR_MOVSD, 3 );
  test_rex_vload_vstore( "rex_mov_MOVSS", &mycode, LIBXSMM_X86_INSTR_MOVSS, 3 );
  test_rex_vload_vstore( "rex_mov_MOVDDUP", &mycode, LIBXSMM_X86_INSTR_MOVDDUP, 1 );

  test_rex_vcompute_2reg_general( "rex_reg_ADDPS", &mycode, LIBXSMM_X86_INSTR_ADDPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MULPS", &mycode, LIBXSMM_X86_INSTR_MULPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SUBPS", &mycode, LIBXSMM_X86_INSTR_SUBPS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ADDSS", &mycode, LIBXSMM_X86_INSTR_ADDSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MULSS", &mycode, LIBXSMM_X86_INSTR_MULSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SUBSS", &mycode, LIBXSMM_X86_INSTR_SUBSS, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SHUFPS", &mycode, LIBXSMM_X86_INSTR_SHUFPS, 0x1 );
  test_rex_vcompute_2reg_general( "rex_reg_ADDPD", &mycode, LIBXSMM_X86_INSTR_ADDPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MULPD", &mycode, LIBXSMM_X86_INSTR_MULPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SUBPD", &mycode, LIBXSMM_X86_INSTR_SUBPD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_ADDSD", &mycode, LIBXSMM_X86_INSTR_ADDSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_MULSD", &mycode, LIBXSMM_X86_INSTR_MULSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SUBSD", &mycode, LIBXSMM_X86_INSTR_SUBSD, 0x0 );
  test_rex_vcompute_2reg_general( "rex_reg_SHUFPD", &mycode, LIBXSMM_X86_INSTR_SHUFPD, 0x1 );

  test_rex_vcompute_mem_1reg_general( "rex_mem_ADDPS", &mycode, LIBXSMM_X86_INSTR_ADDPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MULPS", &mycode, LIBXSMM_X86_INSTR_MULPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SUBPS", &mycode, LIBXSMM_X86_INSTR_SUBPS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ADDSS", &mycode, LIBXSMM_X86_INSTR_ADDSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MULSS", &mycode, LIBXSMM_X86_INSTR_MULSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SUBSS", &mycode, LIBXSMM_X86_INSTR_SUBSS, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SHUFPS", &mycode, LIBXSMM_X86_INSTR_SHUFPS, 0x1 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ADDPD", &mycode, LIBXSMM_X86_INSTR_ADDPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MULPD", &mycode, LIBXSMM_X86_INSTR_MULPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SUBPD", &mycode, LIBXSMM_X86_INSTR_SUBPD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_ADDSD", &mycode, LIBXSMM_X86_INSTR_ADDSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_MULSD", &mycode, LIBXSMM_X86_INSTR_MULSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SUBSD", &mycode, LIBXSMM_X86_INSTR_SUBSD, 0x0 );
  test_rex_vcompute_mem_1reg_general( "rex_mem_SHUFPD", &mycode, LIBXSMM_X86_INSTR_SHUFPD, 0x1 );

  /* test VEX/GP instructions */
  test_alu_reg( "alu_reg_ADDQ", &mycode, LIBXSMM_X86_INSTR_ADDQ );
  test_alu_reg( "alu_reg_SUBQ", &mycode, LIBXSMM_X86_INSTR_SUBQ );
  test_alu_reg( "alu_reg_MOVQ", &mycode, LIBXSMM_X86_INSTR_MOVQ );
  test_alu_reg( "alu_reg_CMPQ", &mycode, LIBXSMM_X86_INSTR_CMPQ );
  test_alu_reg( "alu_reg_AMDQ", &mycode, LIBXSMM_X86_INSTR_ANDQ );
  test_alu_reg( "alu_reg_CMOVA", &mycode, LIBXSMM_X86_INSTR_CMOVA );
  test_alu_reg( "alu_reg_CMOVAE", &mycode, LIBXSMM_X86_INSTR_CMOVAE );
  test_alu_reg( "alu_reg_CMOVB", &mycode, LIBXSMM_X86_INSTR_CMOVB );
  test_alu_reg( "alu_reg_CMOVBE", &mycode, LIBXSMM_X86_INSTR_CMOVBE );
  test_alu_reg( "alu_reg_CMOVC", &mycode, LIBXSMM_X86_INSTR_CMOVC );
  test_alu_reg( "alu_reg_CMOVE", &mycode, LIBXSMM_X86_INSTR_CMOVE );
  test_alu_reg( "alu_reg_CMOVG", &mycode, LIBXSMM_X86_INSTR_CMOVG );
  test_alu_reg( "alu_reg_CMOVGE", &mycode, LIBXSMM_X86_INSTR_CMOVGE );
  test_alu_reg( "alu_reg_CMOVL", &mycode, LIBXSMM_X86_INSTR_CMOVL );
  test_alu_reg( "alu_reg_CMOVLE", &mycode, LIBXSMM_X86_INSTR_CMOVLE );
  test_alu_reg( "alu_reg_CMOVNA", &mycode, LIBXSMM_X86_INSTR_CMOVNA );
  test_alu_reg( "alu_reg_CMOVNAE", &mycode, LIBXSMM_X86_INSTR_CMOVNAE );
  test_alu_reg( "alu_reg_CMOVNB", &mycode, LIBXSMM_X86_INSTR_CMOVNB );
  test_alu_reg( "alu_reg_CMOVNBE", &mycode, LIBXSMM_X86_INSTR_CMOVNBE );
  test_alu_reg( "alu_reg_CMOVNC", &mycode, LIBXSMM_X86_INSTR_CMOVNC );
  test_alu_reg( "alu_reg_CMOVNE", &mycode, LIBXSMM_X86_INSTR_CMOVNE );
  test_alu_reg( "alu_reg_CMOVNG", &mycode, LIBXSMM_X86_INSTR_CMOVNG );
  test_alu_reg( "alu_reg_CMOVNGE", &mycode, LIBXSMM_X86_INSTR_CMOVNGE );
  test_alu_reg( "alu_reg_CMOVNL", &mycode, LIBXSMM_X86_INSTR_CMOVNL );
  test_alu_reg( "alu_reg_CMOVNLE", &mycode, LIBXSMM_X86_INSTR_CMOVNLE );
  test_alu_reg( "alu_reg_CMOVNO", &mycode, LIBXSMM_X86_INSTR_CMOVNO );
  test_alu_reg( "alu_reg_CMOVNP", &mycode, LIBXSMM_X86_INSTR_CMOVNP );
  test_alu_reg( "alu_reg_CMOVNS", &mycode, LIBXSMM_X86_INSTR_CMOVNS );
  test_alu_reg( "alu_reg_CMOVNZ", &mycode, LIBXSMM_X86_INSTR_CMOVNZ );
  test_alu_reg( "alu_reg_CMOVO", &mycode, LIBXSMM_X86_INSTR_CMOVO );
  test_alu_reg( "alu_reg_CMOVP", &mycode, LIBXSMM_X86_INSTR_CMOVP );
  test_alu_reg( "alu_reg_CMOVPE", &mycode, LIBXSMM_X86_INSTR_CMOVPE );
  test_alu_reg( "alu_reg_CMOVPO", &mycode, LIBXSMM_X86_INSTR_CMOVPO );
  test_alu_reg( "alu_reg_CMOVS", &mycode, LIBXSMM_X86_INSTR_CMOVS );
  test_alu_reg( "alu_reg_CMOVZ", &mycode, LIBXSMM_X86_INSTR_CMOVZ );
  test_alu_reg( "alu_reg_POPCNT", &mycode, LIBXSMM_X86_INSTR_POPCNT );
  test_alu_reg( "alu_reg_TZCNT", &mycode, LIBXSMM_X86_INSTR_TZCNT );

  /* test alu mem */
  test_alu_mem( "alu_mov_MOVB_LD", &mycode, LIBXSMM_X86_INSTR_MOVB, 1 );
  test_alu_mem( "alu_mov_MOVB_ST", &mycode, LIBXSMM_X86_INSTR_MOVB, 2 );
  test_alu_mem( "alu_mov_MOVW_LD", &mycode, LIBXSMM_X86_INSTR_MOVW, 1 );
  test_alu_mem( "alu_mov_MOVW_ST", &mycode, LIBXSMM_X86_INSTR_MOVW, 2 );
  test_alu_mem( "alu_mov_MOVL_LD", &mycode, LIBXSMM_X86_INSTR_MOVL, 1 );
  test_alu_mem( "alu_mov_MOVL_ST", &mycode, LIBXSMM_X86_INSTR_MOVL, 2 );
  test_alu_mem( "alu_mov_MOVQ_LD", &mycode, LIBXSMM_X86_INSTR_MOVQ, 1 );
  test_alu_mem( "alu_mov_MOVQ_ST", &mycode, LIBXSMM_X86_INSTR_MOVQ, 2 );

  free( codebuffer );

  return 0;
}
