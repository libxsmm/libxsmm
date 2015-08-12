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

/* TODO change this */
int libxsmm_jit_code = 0;

// This routine is for the jit code. All offsets/displacements have similar
// byte patterns, so this is used for all of them
static inline int add_offset ( int place1, int place2, int offset,
                               int forced, int sizereg, unsigned char *buf )
{
   if ( (offset == 0) && (forced==0) ) return ( 0 );
   else if ( ((offset%sizereg)==0) && (offset/sizereg <= 127) && (offset/sizereg >=-128) )
   {
      buf[place1] += 0x40;
      buf[place2] = offset/sizereg;
      return ( 1 );
   } else {
      unsigned char *cptr = (unsigned char *) &offset;
      buf[place1] += 0x80;
      buf[place2] = cptr[0];
      buf[place2+1] = cptr[1];
      buf[place2+2] = cptr[2];
      buf[place2+3] = cptr[3];
      return ( 4 );
   }
}

void libxsmm_instruction_vec_move( libxsmm_generated_code* io_generated_code, 
                                   const unsigned int      i_instruction_set,
                                   const unsigned int      i_vmove_instr, 
                                   const unsigned int      i_gp_reg_number,
                                   const int               i_displacement,
                                   const char              i_vector_name,
                                   const unsigned int      i_vec_reg_number_0,
                                   const unsigned int      i_use_masking,
                                   const unsigned int      i_is_store ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    int maxsize = io_generated_code->buffer_size;
    int iregnum = i_gp_reg_number % 8;
    int vregnum = i_vec_reg_number_0 % 8;
    int ivectype=0, ivectype2=0, iregoff=0, vregoffset2=0, ivectype3=0;
    int aligned=0, forced_offset=0, penultimate=0;
    int place, num=0, vregoffset=0, num2=0, num3=0, sizereg=1;
    int maskingoff=0;
    int bytes = 4; // base number of bytes

    int i_mask_reg_number = 1; // change if you don't want k1
 
    if ( (i_vector_name != 'z') && (i_use_masking!=0) )
    {
       fprintf(stderr,"Masking is only enabled with zmm registers!\n");
       exit(-1);
    }
    if ( maxsize - i < 20 )
    {
       fprintf(stderr,"Most instructions need at most 20 bytes\n");
       exit(-1);
    }
    num = i_vec_reg_number_0 / 8;
    switch ( i_vmove_instr ) {
       case LIBXSMM_X86_INSTR_VMOVAPD:
          aligned += 0x18;
          if ( i_vector_name=='x' ) ivectype += 1;
          if ( num == 1 ) ivectype3 -= 0x80;
          ivectype2 += 0x81;
          sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVAPS:
          aligned += 0x18;
          if ( num == 1 ) ivectype3 -= 0x80;
          if ( i_vector_name!='x' ) ivectype -= 1; // single
          sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVSS:
          if ( i_vector_name!='x' )
          {
             fprintf(stderr,"You want to use vmovss without xmm? ha!\n");
             exit(-1);
          }
          ivectype += 2;
          break;
       case LIBXSMM_X86_INSTR_VMOVSD:
          if ( i_vector_name!='x' )
          {
             fprintf(stderr,"You want to use vmovsd without xmm? ha!\n");
             exit(-1);
          }
          ivectype += 3;
          break;
       case LIBXSMM_X86_INSTR_VBROADCASTSD:
          bytes = 5;
          if ( i_vector_name=='x' ) 
          {
             fprintf(stderr,"vbroadcastsd and xmm? Fool!\n");
             exit(-1);
          }
          if ( i_is_store == 1 ) 
          {
             fprintf(stderr,"vbroadcastsd and stores? I wish!\n");
             exit(-1);
          }
          ivectype2 += 0x81;
          penultimate += 9;
          num2 += 1;
          num3 += 0x21;
          sizereg = 8;
          break;
       case LIBXSMM_X86_INSTR_VBROADCASTSS:
          if ( i_vector_name=='x' ) 
          {
             fprintf(stderr,"vbroadcastss and xmm? Fool!\n");
             exit(-1);
          }
          if ( i_is_store == 1 ) 
          {
             fprintf(stderr,"vbroadcastss and stores? I wish!\n");
             exit(-1);
          }
          bytes = 5;
          ivectype2 += 0x1;
          penultimate += 8;
          sizereg = 4;
          num2 += 1;
          num3 += 0x21;
          break;
       case LIBXSMM_X86_INSTR_VMOVUPD:
          if ( i_vector_name=='x' ) ivectype += 1;
          if ( num == 1 ) ivectype3 -= 0x80;
          sizereg = 64;
          ivectype2 += 0x81;
          break;
       case LIBXSMM_X86_INSTR_VMOVUPS:
          if ( num == 1 ) ivectype3 -= 0x80;
          if ( i_vector_name!='x' ) ivectype -= 1; // single
          sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVDDUP:
          if ( i_is_store == 1 ) 
          {
             fprintf(stderr,"vmovddup and stores? I wish!\n");
             exit(-1);
          }
          ivectype += 2;
          ivectype2 += 0x83;
          if ( num == 1 ) ivectype3 -= 0x80;
          penultimate += 2;
          sizereg = 64;
          if ( i_vector_name=='x' ) ivectype += 1;
          break;
       default:
          fprintf(stderr,"Are you looney?\n"); 
          exit(-1);
    }
    switch ( i_vector_name ) {
       case 'x':
          sizereg = 1;
          if ( num > 1 ) 
          {
             fprintf(stderr,"Are you sure xmm%d exists?\n",i_vec_reg_number_0);
             exit(-1);
          }
          break;
       case 'y':
          ivectype += 5;
          sizereg = 1;
          if ( num > 2 ) 
          {
             fprintf(stderr,"Are you sure ymm%d exists?\n",i_vec_reg_number_0);
             exit(-1);
          }
          break;
       case 'z':
          bytes = 6;
          break;
       default:
          fprintf(stderr,"Exactly what sort of fp regs are you using?\n");
          exit(-1);
    }
    if ( i_gp_reg_number >= 8 ) 
    {
       if ( bytes < 5 ) bytes = 5;
       else iregoff -= 0x20;
    }
    if ( i_is_store == 1 ) 
    {
       aligned += 1;
       if ( i_use_masking != 0 ) maskingoff = i_mask_reg_number;
    } else {
       if ( i_use_masking != 0 ) maskingoff = 0x80 + i_mask_reg_number;
    }
    if ( num == 0 ) vregoffset = 0x90;
    else if ( num == 1 ) { vregoffset = 0x10; vregoffset2 = -0x80; }
    else if ( num == 2 ) vregoffset = 0x80;
    else if ( num == 3 ) vregoffset = 0x00;
    if ( (iregnum == 5) && (i_displacement==0) ) 
    {
       // Registers like rbp/r13 when you have a displacement of 0, we need
       // force the single byte of zero to appear. 
       forced_offset=1;
    }
 
    if ( bytes == 4 )
    {
       buf[i++] = 0xc5;
       buf[i++] = 0xf8 + ivectype + ivectype3;
    } else if ( bytes == 5 ) {
       buf[i++] = 0xc4;
       buf[i++] = 0xc1 + num3 + vregoffset2 + iregoff;
       buf[i++] = 0x78 + ivectype;
    } else if ( bytes == 6 ) {
       buf[i++] = 0x62;
       buf[i++] = 0x61 + vregoffset + iregoff + num2;
       buf[i++] = 0x7c + ivectype2;
       buf[i++] = 0x48 + maskingoff;
    }
    buf[i++] = 0x10 + aligned + penultimate;
    buf[i++] = 0x00 + iregnum + 8*vregnum;
    place = i-1;
    if ( iregnum == LIBXSMM_X86_GP_REG_RSP ) buf[i++] = 0x24;
    i += add_offset ( place, i, i_displacement, forced_offset, sizereg, buf );
    
    io_generated_code->code_size = i;
    
  } else {
    char l_new_code[512];
    char l_gp_reg_name[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name );
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_vmove_instr, l_instr_name );

    if ( (i_instruction_set == LIBXSMM_X86_AVX512) && (i_use_masking != 0) ) {
      /* build vmovpd/ps/sd/ss instruction, load use */
      if ( i_is_store == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %i(%%%%%s), %%%%%cmm%i%%{%%%%k%i%%}%%{z%%}\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          sprintf(l_new_code, "                       %s %i(%%%s), %%%cmm%i{%%k%i}{z}\n", l_instr_name, i_displacement, l_gp_reg_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %%%%%cmm%i, %i(%%%%%s)%%{%%%%k%i%%}\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          sprintf(l_new_code, "                       %s %%%cmm%i, %i(%%%s){%%k%i}\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      }
    } else if ( (i_instruction_set == LIBXSMM_X86_IMCI) && (i_use_masking != 0) ) {
      /* build vmovpd/ps/sd/ss instruction, load use */
      if ( i_is_store == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %i(%%%%%s), %%%%%cmm%i%%{%%%%k%i%%}\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          sprintf(l_new_code, "                       %s %i(%%%s), %%%cmm%i{%%k%i}\n", l_instr_name, i_displacement, l_gp_reg_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %%%%%cmm%i, %i(%%%%%s)%%{%%%%k%i%%}\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          sprintf(l_new_code, "                       %s %%%cmm%i, %i(%%%s){%%k%i}\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      }
    } else {
      /* build vmovpd/ps/sd/ss instruction, load use */
      if ( i_is_store == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %i(%%%%%s), %%%%%cmm%i\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_name, i_vector_name, i_vec_reg_number_0 );
        } else {
          sprintf(l_new_code, "                       %s %i(%%%s), %%%cmm%i\n", l_instr_name, i_displacement, l_gp_reg_name, i_vector_name, i_vec_reg_number_0 );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          sprintf(l_new_code, "                       \"%s %%%%%cmm%i, %i(%%%%%s)\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_name );
        } else {
          sprintf(l_new_code, "                       %s %%%cmm%i, %i(%%%s)\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_name );
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
                                               const unsigned int      i_displacement,
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
      fprintf(stderr, "LIBXSMM ERROR, libxsmm_instruction_vec_compute_membcast: is not supported on other platforms than AVX512/IMCI\n");
      exit(-1);
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
                                   const unsigned int      i_gp_reg_number,
                                   const int               i_displacement ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_gp_reg_name[4];
    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name );
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_prefetch_instr, l_instr_name );

    if ( io_generated_code->code_type == 0 ) {
      sprintf(l_new_code, "                       \"%s %i(%%%%%s)\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_name );
    } else {
      sprintf(l_new_code, "                       %s %i(%%%s)\n", l_instr_name, i_displacement, l_gp_reg_name );
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

    if ( io_generated_code->code_type == 0 ) {
      sprintf(l_new_code, "                       \"%s %%%%%sd, %%%%k%i\\n\\t\"\n", l_instr_name, l_gp_reg_name, i_mask_reg_number );
    } else { 
      sprintf(l_new_code, "                       %s %%%sd, %%k%i\n", l_instr_name, l_gp_reg_name, i_mask_reg_number );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }
}

void libxsmm_instruction_register_jump_label( libxsmm_generated_code* io_generated_code,
                                              const char*             i_jmp_label ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];

    if ( io_generated_code->code_type == 0 ) {
      sprintf(l_new_code, "                       \"%s:\\n\\t\"\n", i_jmp_label );
    } else {
      sprintf(l_new_code, "                       %s:\n", i_jmp_label );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
  }  
}

void libxsmm_instruction_jump_to_label( libxsmm_generated_code* io_generated_code,
                                        const unsigned int      i_jmp_instr,
                                        const char*             i_jmp_label ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
  } else {
    char l_new_code[512];
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_jmp_instr, l_instr_name );

    if ( io_generated_code->code_type == 0 ) {
      sprintf(l_new_code, "                       \"%s %s\\n\\t\"\n", l_instr_name, i_jmp_label );
    } else {
      sprintf(l_new_code, "                       %s %s\n", l_instr_name, i_jmp_label );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code );
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
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_close_instruction_stream, reg_a cannot by callee save since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value!\n");
      exit(-1);
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b ) ) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_close_instruction_stream, reg_b cannot by callee save since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value!\n");
      exit(-1);
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_c ) ) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_close_instruction_stream, reg_c cannot by callee save since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value!\n");
      exit(-1);
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a_prefetch ) ) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_close_instruction_stream, reg_pra_a cannot by callee save since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value!\n");
      exit(-1);
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b_prefetch ) ) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_close_instruction_stream, reg_pra_b cannot by callee save since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value!\n");
      exit(-1);
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
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_close_instruction_stream, reg_pra_b cannot by callee save since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value!\n");
      exit(-1);
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a_prefetch ) ) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_close_instruction_stream, reg_pra_a cannot by callee save since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value!\n");
      exit(-1);
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_c ) ) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_close_instruction_stream, reg_c cannot by callee save since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value!\n");
      exit(-1);
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b ) ) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_close_instruction_stream, reg_b cannot by callee save since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value!\n");
      exit(-1);
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a ) ) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_generator_dense_sse_avx_close_instruction_stream, reg_a cannot by callee save since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value!\n");
      exit(-1);
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

