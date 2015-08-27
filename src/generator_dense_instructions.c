/******************************************************************************
** Copyright (c) 2015, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#include "generator_common.h"
#include "generator_dense_instructions.h"

void libxsmm_instruction_vec_move( libxsmm_generated_code* io_generated_code, 
                                   const unsigned int      i_instruction_set,
                                   const unsigned int      i_vmove_instr, 
                                   const unsigned int      i_gp_reg_base,
                                   const unsigned int      i_gp_reg_idx,
                                   const unsigned int      i_scale,
                                   const int               i_displacement,
                                   const char              i_vector_name,
                                   const unsigned int      i_vec_reg_number_0,
                                   const unsigned int      i_use_masking,
                                   const unsigned int      i_is_store ) {
#ifndef NDEBUG
  if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_NO_INDEX_SCALE_ADDR );
    return;
  }
#endif
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_gp_reg_base_name[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base_name );
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_vmove_instr, l_instr_name );

    if ( (i_instruction_set == LIBXSMM_X86_AVX512) && (i_use_masking != 0) ) {
      /* build vmovpd/ps/sd/ss instruction, load use */
      if ( i_is_store == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %i(%%%%%s), %%%%%cmm%i%%{%%%%k%i%%}%%{z%%}\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          sprintf(l_new_code, "                       %s %i(%%%s), %%%cmm%i{%%k%i}{z}\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %%%%%cmm%i, %i(%%%%%s)%%{%%%%k%i%%}\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          sprintf(l_new_code, "                       %s %%%cmm%i, %i(%%%s){%%k%i}\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      }
    } else if ( (i_instruction_set == LIBXSMM_X86_IMCI) && (i_use_masking != 0) ) {
      /* build vmovpd/ps/sd/ss instruction, load use */
      if ( i_is_store == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %i(%%%%%s), %%%%%cmm%i%%{%%%%k%i%%}\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          sprintf(l_new_code, "                       %s %i(%%%s), %%%cmm%i{%%k%i}\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %%%%%cmm%i, %i(%%%%%s)%%{%%%%k%i%%}\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          sprintf(l_new_code, "                       %s %%%cmm%i, %i(%%%s){%%k%i}\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      }
    } else {
      /* build vmovpd/ps/sd/ss instruction, load use */
      if ( i_is_store == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %i(%%%%%s), %%%%%cmm%i\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0 );
        } else {
          sprintf(l_new_code, "                       %s %i(%%%s), %%%cmm%i\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0 );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %%%%%cmm%i, %i(%%%%%s)\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name );
        } else {
          sprintf(l_new_code, "                       %s %%%cmm%i, %i(%%%s)\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name );
        }
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_vec_compute_reg( libxsmm_generated_code* io_generated_code, 
                                          const unsigned int      i_instruction_set,
                                          const unsigned int      i_vec_instr,
                                          const char              i_vector_name,                                
                                          const unsigned int      i_vec_reg_number_0,
                                          const unsigned int      i_vec_reg_number_1,
                                          const unsigned int      i_vec_reg_number_2 ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name );

    /* build vXYZpd/ps/sd/ss instruction pure register use*/
    if ( i_instruction_set != LIBXSMM_X86_SSE3 ) {
      if ( io_generated_code->code_type == 0 ) {
        sprintf(l_new_code, "                       \"%s %%%%%cmm%i, %%%%%cmm%i, %%%%%cmm%i\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2 );
      } else {
        sprintf(l_new_code, "                       %s %%%cmm%i, %%%cmm%i, %%%cmm%i\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2 );
      }
    } else {
      if ( io_generated_code->code_type == 0 ) {
        sprintf(l_new_code, "                       \"%s %%%%%cmm%i, %%%%%cmm%i\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1);
      } else {
        sprintf(l_new_code, "                       %s %%%cmm%i, %%%cmm%i\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_vec_compute_membcast( libxsmm_generated_code* io_generated_code, 
                                               const unsigned int      i_instruction_set,
                                               const unsigned int      i_vec_instr,
                                               const unsigned int      i_gp_reg_base,
                                               const unsigned int      i_gp_reg_idx,
                                               const unsigned int      i_scale,
                                               const int               i_displacement,
                                               const char              i_vector_name,                                
                                               const unsigned int      i_vec_reg_number_0,
                                               const unsigned int      i_vec_reg_number_1 ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_gp_reg_base[4];
    char l_gp_reg_idx[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base );
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name );
    char l_broadcast[8];
    unsigned int l_single_precision = libxsmm_is_x86_vec_instr_single_precision( i_vec_instr );
    if (l_single_precision == 0) {
      sprintf( l_broadcast, "1to8" );
    } else {
      sprintf( l_broadcast, "1to16" );
    }

    /* build vXYZpd/ps/sd/ss instruction pure register use*/
    if ( i_instruction_set == LIBXSMM_X86_AVX512 || i_instruction_set == LIBXSMM_X86_IMCI ) {
      /* we just a have displacement */
      if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF ) {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %i(%%%%%s)%%{%s%%}, %%%%%cmm%i, %%%%%cmm%i\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        } else {
          sprintf(l_new_code, "                       %s %i(%%%s){%s}, %%%cmm%i, %%%cmm%i\n", l_instr_name, i_displacement, l_gp_reg_base, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        }
      } else {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx );
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %i(%%%%%s,%%%%%s,%i)%%{%s%%}, %%%%%cmm%i, %%%%%cmm%i\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        } else {
          sprintf(l_new_code, "                       %s %i(%%%s,%%%s,%i){%s}, %%%cmm%i, %%%cmm%i\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        }
      }
    } else {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_NO_IMCI_AVX512_BCAST );
      return;
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
}


void libxsmm_instruction_vec_shuffle_reg( libxsmm_generated_code* io_generated_code,
                                          const unsigned int      i_instruction_set, 
                                          const unsigned int      i_vec_instr,
                                          const char              i_vector_name,                                
                                          const unsigned int      i_vec_reg_number_0,
                                          const unsigned int      i_vec_reg_number_1,
                                          const unsigned int      i_vec_reg_number_2,
                                          const unsigned int      i_shuffle_operand ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name );

    if ( i_instruction_set != LIBXSMM_X86_SSE3 ) {
      if ( io_generated_code->code_type == 0 ) {
        sprintf(l_new_code, "                       \"%s $%i, %%%%%cmm%i, %%%%%cmm%i, %%%%%cmm%i\\n\\t\"\n", l_instr_name, i_shuffle_operand, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2 );
      } else {
        sprintf(l_new_code, "                       %s $%i, %%%cmm%i, %%%cmm%i, %%%cmm%i\n", l_instr_name, i_shuffle_operand, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2 );
      }
    } else {
      if ( io_generated_code->code_type == 0 ) {
        sprintf(l_new_code, "                       \"%s $%i, %%%%%cmm%i, %%%%%cmm%i\\n\\t\"\n", l_instr_name, i_shuffle_operand, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
      } else {
        sprintf(l_new_code, "                       %s $%i, %%%cmm%i, %%%cmm%i\n", l_instr_name, i_shuffle_operand, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_prefetch( libxsmm_generated_code* io_generated_code,
                                   const unsigned int      i_prefetch_instr, 
                                   const unsigned int      i_gp_reg_base,
                                   const unsigned int      i_gp_reg_idx,
                                   const unsigned int      i_scale,
                                   const int               i_displacement ) {
#ifndef NDEBUG
  if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_NO_INDEX_SCALE_ADDR );
    return;
  }
#endif
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_gp_reg_base_name[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base_name );
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_prefetch_instr, l_instr_name );

    if ( io_generated_code->code_type == 0 ) {
      sprintf(l_new_code, "                       \"%s %i(%%%%%s)\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name );
    } else {
      sprintf(l_new_code, "                       %s %i(%%%s)\n", l_instr_name, i_displacement, l_gp_reg_base_name );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_alu_imm( libxsmm_generated_code* io_generated_code,
                                  const unsigned int      i_alu_instr,
                                  const unsigned int      i_gp_reg_number,
                                  const unsigned int      i_immediate ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_gp_reg_name[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name );
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_alu_instr, l_instr_name );

    if ( io_generated_code->code_type == 0 ) {
      sprintf(l_new_code, "                       \"%s $%i, %%%%%s\\n\\t\"\n", l_instr_name, i_immediate, l_gp_reg_name );
    } else { 
      sprintf(l_new_code, "                       %s $%i, %%%s\n", l_instr_name, i_immediate, l_gp_reg_name );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_alu_reg( libxsmm_generated_code* io_generated_code,
                                  const unsigned int      i_alu_instr,
                                  const unsigned int      i_gp_reg_number_src,
                                  const unsigned int      i_gp_reg_number_dest) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_gp_reg_name_src[4];
    char l_gp_reg_name_dest[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_number_src, l_gp_reg_name_src );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_number_dest, l_gp_reg_name_dest );
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_alu_instr, l_instr_name );

    if ( io_generated_code->code_type == 0 ) {
      sprintf(l_new_code, "                       \"%s %%%%%s, %%%%%s\\n\\t\"\n", l_instr_name, l_gp_reg_name_src, l_gp_reg_name_dest );
    } else { 
      sprintf(l_new_code, "                       %s %%%s, %%%s\n", l_instr_name, l_gp_reg_name_src, l_gp_reg_name_dest );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_mask_move( libxsmm_generated_code* io_generated_code,
                                    const unsigned int      i_mask_instr,
                                    const unsigned int      i_gp_reg_number,
                                    const unsigned int      i_mask_reg_number ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_gp_reg_name[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name );
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_mask_instr, l_instr_name );
    char l_prefix = '\0';

    /* check if we need to add a prefix for accessing 32bit in a 64bit register */
    if ( i_gp_reg_number == LIBXSMM_X86_GP_REG_R8  ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R9  ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R10 ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R11 ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R12 ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R13 ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R14 ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R15    ) {
      l_prefix = 'd';
    }

    if ( io_generated_code->code_type == 0 ) {
      sprintf(l_new_code, "                       \"%s %%%%%s%c, %%%%k%i\\n\\t\"\n", l_instr_name, l_gp_reg_name, l_prefix, i_mask_reg_number );
    } else { 
      sprintf(l_new_code, "                       %s %%%s%c, %%k%i\n", l_instr_name, l_gp_reg_name, l_prefix, i_mask_reg_number );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_register_jump_label( libxsmm_generated_code*     io_generated_code,
                                              libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  /* check if we still have lable we can jump to */
  if ( io_loop_label_tracker->label_count == 32 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    
    io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] = io_loop_label_tracker->label_count;

    if ( io_generated_code->code_type == 0 ) {
      sprintf(l_new_code, "                       \"%i:\\n\\t\"\n", io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] );
    } else {
      sprintf(l_new_code, "                       %i:\n", io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
    io_loop_label_tracker->label_count++;
  }  
}

void libxsmm_instruction_jump_back_to_label( libxsmm_generated_code*     io_generated_code,
                                             const unsigned int          i_jmp_instr,
                                             libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  /* check that we just handle jl */
  if ( i_jmp_instr != LIBXSMM_X86_INSTR_JL) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_UNSUPPORTED_JUMP );
    return;
  }

  /* check if we still have lable we can jump to */
  if ( io_loop_label_tracker->label_count == 0 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_NO_JMPLBL_AVAIL );
    return;
  }

  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_jmp_instr, l_instr_name );
    
    io_loop_label_tracker->label_count--;
    
    if ( io_generated_code->code_type == 0 ) {
      sprintf(l_new_code, "                       \"%s %ib\\n\\t\"\n", l_instr_name, io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] );
    } else {
      sprintf(l_new_code, "                       %s %ib\n", l_instr_name, io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
    
    io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] = 0;
  }
}

void libxsmm_generator_dense_x86_open_instruction_stream( libxsmm_generated_code*       io_generated_code,
                                                           const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                                           const char*                   i_arch, 
                                                           const char*                   i_prefetch) { 
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
    /* @TODO-GREG: how do we interface here? */
    /* this is start of the xGEMM kernel, the registers are in the variables */
  } else if ( io_generated_code->code_type == 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    char l_gp_reg_name[4];

    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a ) ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b ) ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_c ) ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_C );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a_prefetch ) ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A_PREF );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b_prefetch ) ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B_PREF );
      return;
    }
    if ( (strcmp(i_arch, "wsm") == 0) ||
         (strcmp(i_arch, "snb") == 0) ||
         (strcmp(i_arch, "hsw") == 0)    ) {
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_mloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_mloop, l_gp_reg_name );
        sprintf( l_new_code, "                       pushq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
      }
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_nloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_nloop, l_gp_reg_name );
        sprintf( l_new_code, "                       pushq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
      }
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_kloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_name );
        sprintf( l_new_code, "                       pushq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
      }
    } else {
      libxsmm_append_code_as_string( io_generated_code, "                       pushq %rbx\n" );
      libxsmm_append_code_as_string( io_generated_code, "                       pushq %r12\n" );
      libxsmm_append_code_as_string( io_generated_code, "                       pushq %r13\n" );
      libxsmm_append_code_as_string( io_generated_code, "                       pushq %r14\n" );
      libxsmm_append_code_as_string( io_generated_code, "                       pushq %r15\n" );
    }
  } else {
    char l_new_code[512];
    char l_gp_reg_name[4];
    
    /* loading b pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b, l_gp_reg_name );
    sprintf( l_new_code, "  __asm__ __volatile__(\"movq %%0, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code );

    /* loading a pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a, l_gp_reg_name );
    sprintf( l_new_code, "                       \"movq %%1, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code );

    /* loading c pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_c, l_gp_reg_name );
    sprintf( l_new_code, "                       \"movq %%2, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code );

    /* loading b prefetch pointer in assembly */
    if ( ( strcmp(i_prefetch, "BL2viaC") == 0 ) || 
         ( strcmp(i_prefetch, "curAL2_BL2viaC") == 0 )    ) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_name );
      sprintf( l_new_code, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_code_as_string( io_generated_code, l_new_code );
    /* loading a prefetch pointer in assembly */
    } else if ( ( strcmp(i_prefetch, "AL2jpst") == 0 ) ||
                ( strcmp(i_prefetch, "AL2") == 0 )        ) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_name );
      sprintf( l_new_code, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_code_as_string( io_generated_code, l_new_code );
    /* loading a and b prefetch pointer in assembly */
    } else if ( ( strcmp(i_prefetch, "AL2jpst_BL2viaC") == 0 ) ||
                ( strcmp(i_prefetch, "AL2_BL2viaC") == 0 )        ) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_name );
      sprintf( l_new_code, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_code_as_string( io_generated_code, l_new_code );
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_name );
      sprintf( l_new_code, "                       \"movq %%4, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_code_as_string( io_generated_code, l_new_code );
    } else {}
  }

  /* reset loop counters */
  libxsmm_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_mloop, 0 );
  libxsmm_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_nloop, 0 );
  libxsmm_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_kloop, 0 );
}

void libxsmm_generator_dense_x86_close_instruction_stream( libxsmm_generated_code*       io_generated_code,
                                                           const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                                           const char*                   i_arch, 
                                                           const char*                   i_prefetch) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
    /* @TODO-GREG: how do we interface here? */
    /* this is start of the xGEMM kernel, the registers are in the variables */
  } else if ( io_generated_code->code_type == 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    char l_gp_reg_name[4];

    if ( (strcmp(i_arch, "wsm") == 0) ||
         (strcmp(i_arch, "snb") == 0) ||
         (strcmp(i_arch, "hsw") == 0)    ) {
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_kloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_name );
        sprintf( l_new_code, "                       popq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
      }
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_nloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_nloop, l_gp_reg_name );
        sprintf( l_new_code, "                       popq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
      }
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_mloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_mloop, l_gp_reg_name );
        sprintf( l_new_code, "                       popq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code );
      }
    } else {
      libxsmm_append_code_as_string( io_generated_code, "                       popq %r15\n" );
      libxsmm_append_code_as_string( io_generated_code, "                       popq %r14\n" );
      libxsmm_append_code_as_string( io_generated_code, "                       popq %r13\n" );
      libxsmm_append_code_as_string( io_generated_code, "                       popq %r12\n" );
      libxsmm_append_code_as_string( io_generated_code, "                       popq %rbx\n" );
    }

    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b_prefetch ) ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B_PREF );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a_prefetch ) ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A_PREF );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_c ) ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_C );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b ) ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a ) ) {
      libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A );
      return;
    }
    /* @TODO: I don't know if this is the correct placement in the generation process */
    libxsmm_append_code_as_string( io_generated_code, "                       retq\n" );
  } else {
    char l_new_code[1024];
    char l_gp_reg_a[4];
    char l_gp_reg_b[4];
    char l_gp_reg_c[4];
    char l_gp_reg_pre_a[4];
    char l_gp_reg_pre_b[4];
    char l_gp_reg_mloop[4];
    char l_gp_reg_nloop[4];
    char l_gp_reg_kloop[4];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a, l_gp_reg_a );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b, l_gp_reg_b );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_c, l_gp_reg_c );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_pre_a );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_pre_b );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_mloop, l_gp_reg_mloop );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_nloop, l_gp_reg_nloop );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_kloop );

    if ( ( strcmp(i_prefetch, "BL2viaC") == 0 ) || 
         ( strcmp(i_prefetch, "curAL2_BL2viaC") == 0 )    ) {
      if ( (strcmp(i_arch, "wsm") == 0) ||
           (strcmp(i_arch, "snb") == 0) ||
           (strcmp(i_arch, "hsw") == 0)    ) {
        sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(B_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_b, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(B_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else if ( ( strcmp(i_prefetch, "AL2jpst") == 0 ) ||
                ( strcmp(i_prefetch, "AL2") == 0 )        ) {
      if ( (strcmp(i_arch, "wsm") == 0) ||
           (strcmp(i_arch, "snb") == 0) ||
           (strcmp(i_arch, "hsw") == 0)    ) {
        sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else if ( ( strcmp(i_prefetch, "AL2jpst_BL2viaC") == 0 ) ||
                ( strcmp(i_prefetch, "AL2_BL2viaC") == 0 )        ) {
      if ( (strcmp(i_arch, "wsm") == 0) ||
           (strcmp(i_arch, "snb") == 0) ||
           (strcmp(i_arch, "hsw") == 0)    ) {
        sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch), \"m\"(B_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_pre_b, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch), \"m\"(B_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else {
      if ( (strcmp(i_arch, "wsm") == 0) ||
           (strcmp(i_arch, "snb") == 0) ||
           (strcmp(i_arch, "hsw") == 0)    ) {
        sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
}

