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

// This routine is for the jit code. All offsets/displacements have similar
// byte patterns, so this is used for all of them
static inline int add_offset ( const unsigned int i_place1,
                               const unsigned int i_place2,
                               const int i_offset,
                               const unsigned int i_forced,
                               const int i_sizereg,
                               unsigned char *buf )
{
   if ( (i_offset == 0) && (i_forced==0) ) return ( 0 );
   else if ( ((i_offset%i_sizereg)==0) &&
              (i_offset/i_sizereg <= 127) &&
              (i_offset/i_sizereg >=-128) )
   {
      buf[i_place1] += 0x40;
      buf[i_place2] = i_offset/i_sizereg;
      return ( 1 );
   } else {
      unsigned char *l_cptr = (unsigned char *) &i_offset;
      buf[ i_place1 ] += 0x80;
      buf[ i_place2 ] = l_cptr[0];
      buf[i_place2+1] = l_cptr[1];
      buf[i_place2+2] = l_cptr[2];
      buf[i_place2+3] = l_cptr[3];
      return ( 4 );
   }
}

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
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int l_iregnum = i_gp_reg_base % 8;
    int l_vregnum = i_vec_reg_number_0 % 8;
    int l_ivectype=0, l_ivectype2=0, l_iregoff=0, l_ivectype3=0;
    int l_vregoffset=0, l_vregoffset2=0;
    int l_aligned=0, l_forced_offset=0, l_penultimate=0;
    int l_place, l_num=0, l_num2=0, l_num3=0, l_sizereg=1;
    int l_maskingoff=0;
    int l_bytes = 4; // base number of bytes

    int i_mask_reg_number = 1; // change if you don't want k1
 
    if ( (i_vector_name != 'z') && (i_use_masking!=0) )
    {
       fprintf(stderr,"Masking is only enabled with zmm registers!\n");
       exit(-1);
    }
    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"Most instructions need at most 20 bytes\n");
       exit(-1);
    }
    l_num = i_vec_reg_number_0 / 8;
    switch ( i_vmove_instr ) {
       case LIBXSMM_X86_INSTR_VMOVAPD:
          l_aligned += 0x18;
          if ( i_vector_name=='x' ) l_ivectype += 1;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_ivectype2 += 0x81;
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVAPS:
          l_aligned += 0x18;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          if ( i_vector_name!='x' ) l_ivectype -= 1; // single
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVSS:
          if ( i_vector_name!='x' )
          {
             fprintf(stderr,"You want to use vmovss without xmm? ha!\n");
             exit(-1);
          }
          l_ivectype += 2;
          break;
       case LIBXSMM_X86_INSTR_VMOVSD:
          if ( i_vector_name!='x' )
          {
             fprintf(stderr,"You want to use vmovsd without xmm? ha!\n");
             exit(-1);
          }
          l_ivectype += 3;
          break;
       case LIBXSMM_X86_INSTR_VBROADCASTSD:
          l_bytes = 5;
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
          l_ivectype2 += 0x81;
          l_penultimate += 9;
          l_num2 += 1;
          l_num3 += 0x21;
          l_sizereg = 8;
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
          l_bytes = 5;
          l_ivectype2 += 0x1;
          l_penultimate += 8;
          l_sizereg = 4;
          l_num2 += 1;
          l_num3 += 0x21;
          break;
       case LIBXSMM_X86_INSTR_VMOVUPD:
          if ( i_vector_name=='x' ) l_ivectype += 1;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_sizereg = 64;
          l_ivectype2 += 0x81;
          break;
       case LIBXSMM_X86_INSTR_VMOVUPS:
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          if ( i_vector_name!='x' ) l_ivectype -= 1; // single
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVDDUP:
          if ( i_is_store == 1 ) 
          {
             fprintf(stderr,"vmovddup and stores? I wish!\n");
             exit(-1);
          }
          l_ivectype += 2;
          l_ivectype2 += 0x83;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_penultimate += 2;
          l_sizereg = 64;
          if ( i_vector_name=='x' ) l_ivectype += 1;
          break;
       default:
          fprintf(stderr,"Are you looney?\n"); 
          exit(-1);
    }
    switch ( i_vector_name ) {
       case 'x':
          l_sizereg = 1;
          if ( l_num > 1 ) 
          {
             fprintf(stderr,"Are you sure xmm%d exists?\n",i_vec_reg_number_0);
             exit(-1);
          }
          break;
       case 'y':
          l_ivectype += 5;
          l_sizereg = 1;
          if ( l_num > 2 ) 
          {
             fprintf(stderr,"Are you sure ymm%d exists?\n",i_vec_reg_number_0);
             exit(-1);
          }
          break;
       case 'z':
          l_bytes = 6;
          break;
       default:
          fprintf(stderr,"Exactly what sort of fp regs are you using?\n");
          exit(-1);
    }
    if ( i_gp_reg_base >= 8 ) 
    {
       if ( l_bytes < 5 ) l_bytes = 5;
       else l_iregoff -= 0x20;
    }
    if ( i_is_store == 1 ) 
    {
       l_aligned += 1;
       if ( i_use_masking != 0 ) l_maskingoff = i_mask_reg_number;
    } else {
       if ( i_use_masking != 0 ) l_maskingoff = 0x80 + i_mask_reg_number;
    }
    if ( l_num == 0 ) l_vregoffset = 0x90;
    else if ( l_num == 1 ) { l_vregoffset = 0x10; l_vregoffset2 = -0x80; }
    else if ( l_num == 2 ) l_vregoffset = 0x80;
    else if ( l_num == 3 ) l_vregoffset = 0x00;
    if ( (l_iregnum == 5) && (i_displacement==0) ) 
    {
       // Registers like rbp/r13 when you have a displacement of 0, we need
       // force the single byte of zero to appear. 
       l_forced_offset=1;
    }
 
    if ( l_bytes == 4 )
    {
       buf[i++] = 0xc5;
       buf[i++] = 0xf8 + l_ivectype + l_ivectype3;
    } else if ( l_bytes == 5 ) {
       buf[i++] = 0xc4;
       buf[i++] = 0xc1 + l_num3 + l_vregoffset2 + l_iregoff;
       buf[i++] = 0x78 + l_ivectype;
    } else if ( l_bytes == 6 ) {
       buf[i++] = 0x62;
       buf[i++] = 0x61 + l_vregoffset + l_iregoff + l_num2;
       buf[i++] = 0x7c + l_ivectype2;
       buf[i++] = 0x48 + l_maskingoff;
    }
    buf[i++] = 0x10 + l_aligned + l_penultimate;
    buf[i++] = 0x00 + l_iregnum + 8*l_vregnum;
    l_place = i-1;
    if ( l_iregnum == LIBXSMM_X86_GP_REG_RSP ) buf[i++] = 0x24;
    i += add_offset ( l_place, i, i_displacement, l_forced_offset, l_sizereg, buf );
    
    io_generated_code->code_size = i;
    
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
                                          const unsigned int      i_vec_reg_number_2 ) 
{
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    // int i = *loc;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    // unsigned int l_maxsize = 1024;
    int l_second=0, l_third=0, l_fourth=0, l_xreg=0;
    int l_reg0   = i_vec_reg_number_0;
    int l_reg1   = i_vec_reg_number_1; 
    int l_reg2   = i_vec_reg_number_2;
    int l_fpadj=0;
    int l_fpadj2=0;
    int l_bytes=4;

    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"Most instructions need at most 20 bytes\n");
       exit(-1);
    }
    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_VXORPD:
          l_fpadj = -2;
          break;
       case LIBXSMM_X86_INSTR_VMULPD:
          break;
       case LIBXSMM_X86_INSTR_VADDPD:
          l_fpadj = -1;
          break;
       case LIBXSMM_X86_INSTR_VSUBPD:
          l_fpadj = 3;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231PD:
          l_second += 0x21;
          l_fpadj  += 0x5f;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231PD:
          l_second += 0x21;
          l_fpadj  += 0x61;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231PD:
          l_second += 0x21;
          l_fpadj  += 0x63;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231PD:
          l_second += 0x21;
          l_fpadj  += 0x65;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VMULSD:
          l_fpadj2 = 2; 
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VADDSD:
          l_fpadj  =-1;
          l_fpadj2 = 2;
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VSUBSD:
          l_fpadj  = 3;
          l_fpadj2 = 2;
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SD:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x60;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231SD:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x62;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231SD:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x64;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231SD:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x66;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VXORPS:
          l_fpadj2 = -1;
          l_fpadj = -2;
          if ( i_vector_name == 'z' )
          {
             l_fpadj2 -= 0x80;
          }
          break;
       case LIBXSMM_X86_INSTR_VMULPS:
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) && 
               (i_vec_reg_number_1<=15) && (i_vec_reg_number_2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          break;
       case LIBXSMM_X86_INSTR_VADDPS:
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) && 
               (i_vec_reg_number_1<=15) && (i_vec_reg_number_2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = -1;
          break;
       case LIBXSMM_X86_INSTR_VSUBPS:
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) && 
               (i_vec_reg_number_1<=15) && (i_vec_reg_number_2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = 3;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231PS:
          l_second += 0x21;
          l_fpadj  += 0x5f;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231PS:
          l_second += 0x21;
          l_fpadj  += 0x61;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231PS:
          l_second += 0x21;
          l_fpadj  += 0x63;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231PS:
          l_second += 0x21;
          l_fpadj  += 0x65;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VMULSS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_fpadj2 = 1; 
          break;
       case LIBXSMM_X86_INSTR_VADDSS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_fpadj  =-1;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VSUBSS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_fpadj  = 3;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x60;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231SS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x62;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231SS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x64;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231SS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x66;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_vec_reg_number_0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VPXORD:
          l_bytes = 6;
          if ( i_vector_name == 'x' )
          {
             l_fourth -= 0x40;
          } else if ( i_vector_name == 'y' )
          {
             l_fourth -= 0x20;
          }
          l_fpadj += 0x96;
          l_fpadj2 += 0x80;
          break;
       default:
          fprintf(stderr,"WTF! what are you doing?\n");
          break;
    }
    if ( i_vector_name == 'x' ) l_xreg = -4;
    l_reg0 = i_vec_reg_number_0 % 8;
    l_reg1 = i_vec_reg_number_1 % 8;
    l_reg2 = i_vec_reg_number_2 % 8;
    if ( i_vec_reg_number_2 >= 8 ) { l_second -= 0x80; }
    if ( i_vec_reg_number_1 >= 8 ) { l_third  -= 0x40; }
    if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) && 
         (i_vec_reg_number_1<=15) && (i_vec_reg_number_2<=15) )
    {
       if ( i_vec_reg_number_0 >= 8 ) 
       {
          if ( l_bytes < 5 ) l_bytes = 5;
       }
    } else l_bytes = 6;
 
    if ( l_bytes == 4 )
    {
       buf[i++] = 0xc5;
       buf[i++] = 0xfd - 8*l_reg1   + l_third + l_second + l_xreg + l_fpadj2;
       buf[i++] = 0x59 + l_fpadj;
       buf[i++] = 0xc0 + l_reg0    + 8*l_reg2 ;
    } else if ( l_bytes == 5 )
    {
       buf[i++] = 0xc4;
       buf[i++] = 0xc1 + l_second;
       buf[i++] = 0x7d - 8*l_reg1   + l_third + l_xreg + l_fpadj2;
       buf[i++] = 0x59 + l_fpadj;
       buf[i++] = 0xc0 + l_reg0    + 8*l_reg2 ;
    } else if ( l_bytes == 6 )
    {
       if ( i_vec_reg_number_0 >= 8 ) { l_second -= 0x20; }
       if ( i_vec_reg_number_0 >= 16 ) 
       { 
          l_second -= 0x20; 
          if ( i_vector_name=='x' ) l_fourth -= 0x40;
          if ( i_vector_name=='y' ) l_fourth -= 0x20;
       }
       if ( i_vec_reg_number_0 >= 24 ) { l_second -= 0x20; }
       if ( i_vec_reg_number_1 >= 16 ) 
       { 
          l_third += 0x40; 
          l_fourth -= 0x08; 
          if ( i_vector_name=='x' ) l_fourth -= 0x40;
          if ( i_vector_name=='y' ) l_fourth -= 0x20;
       }
       if ( i_vec_reg_number_1 >= 24 ) { l_third -= 0x40; }
       if ( i_vec_reg_number_2 >= 16 ) { l_second += 0x70; }
       if ( i_vec_reg_number_2 >= 24 ) { l_second -= 0x80; }
       buf[i++] = 0x62;
       buf[i++] = 0xf1 + l_second;
       buf[i++] = 0xfd - 8*l_reg1   + l_third + l_fpadj2;
       buf[i++] = 0x48 + l_fourth;
       buf[i++] = 0x59 + l_fpadj;
       buf[i++] = 0xc0 + l_reg0    + 8*l_reg2 ;
    }

    io_generated_code->code_size = i;
    // *loc = i;

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

// i_displacement should be just const int!!!
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
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    // int i = *loc;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    // unsigned int l_maxsize = 1024;
    int l_second=0, l_third=0, l_fourth=0, l_xreg=0;
    int l_reg0 = 0;
    int l_reg1   = i_vec_reg_number_0; 
    int l_reg2   = i_vec_reg_number_1; 
    int l_fpadj=0, l_place=0;
    int l_fpadj2=0;
    int l_bytes=4;
    int l_forced_offset=0;
    int l_sizereg=64;
    int l_iregoff = 0;

    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"Most instructions need at most 20 bytes\n");
       exit(-1);
    }
    switch ( i_vector_name ) {
       case 'x':
          l_sizereg = 1;
          break;
       case 'y':
          l_sizereg = 1;
          break;
       case 'z':
          l_bytes = 6;
          break;
       default:
          fprintf(stderr,"Exactly what sort of fp regs are you using?\n");
          exit(-1);
    }
    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_VXORPD:
          l_fpadj = -2;
          break;
       case LIBXSMM_X86_INSTR_VMULPD:
          break;
       case LIBXSMM_X86_INSTR_VADDPD:
          l_fpadj = -1;
          break;
       case LIBXSMM_X86_INSTR_VSUBPD:
          l_fpadj = 3;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231PD:
          l_second += 0x21;
          l_fpadj  += 0x5f;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231PD:
          l_second += 0x21;
          l_fpadj  += 0x61;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231PD:
          l_second += 0x21;
          l_fpadj  += 0x63;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231PD:
          l_second += 0x21;
          l_fpadj  += 0x65;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VMULSD:
          l_fpadj2 = 2; 
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VADDSD:
          l_fpadj  =-1;
          l_fpadj2 = 2;
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VSUBSD:
          l_fpadj  = 3;
          l_fpadj2 = 2;
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SD:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x60;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231SD:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x62;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231SD:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x64;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231SD:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x66;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VXORPS:
          l_fpadj2 = -1;
          l_fpadj = -2;
          if ( i_vector_name == 'z' )
          {
             l_fpadj2 -= 0x80;
          }
          break;
       case LIBXSMM_X86_INSTR_VMULPS:
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) && 
               (i_vec_reg_number_1<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          break;
       case LIBXSMM_X86_INSTR_VADDPS:
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) && 
               (i_vec_reg_number_1<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = -1;
          break;
       case LIBXSMM_X86_INSTR_VSUBPS:
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) && 
               (i_vec_reg_number_1<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = 3;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231PS:
          l_second += 0x21;
          l_fpadj  += 0x5f;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231PS:
          l_second += 0x21;
          l_fpadj  += 0x61;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231PS:
          l_second += 0x21;
          l_fpadj  += 0x63;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231PS:
          l_second += 0x21;
          l_fpadj  += 0x65;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VMULSS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_fpadj2 = 1; 
          break;
       case LIBXSMM_X86_INSTR_VADDSS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_fpadj  =-1;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VSUBSS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_fpadj  = 3;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x60;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231SS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x62;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231SS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x64;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231SS:
          if (i_vector_name != 'x') fprintf(stderr,"Really? SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x66;
          if ( i_vector_name == 'z' ) 
          {  
             l_second -= 0x20; 
             l_fpadj2 -= 0x80; 
          } else if ( i_gp_reg_base      > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VPXORD:
          l_bytes = 6;
          if ( i_vector_name == 'x' )
          {
             l_fourth -= 0x40;
             l_sizereg = 16;
          } else if ( i_vector_name == 'y' )
          {
             l_fourth -= 0x20;
             l_sizereg = 32;
          }
          l_fpadj += 0x96;
          l_fpadj2 += 0x80;
          break;
       default:
          fprintf(stderr,"WTF! what are you doing?\n");
          break;
    }
    if ( i_gp_reg_base >= 8 )
    {
       if ( l_bytes < 5 ) l_bytes = 5;
       // else l_iregoff -= 0x20;
    }
    if ( i_vector_name == 'x' ) l_xreg = -4;
    l_reg0 = i_gp_reg_base % 8 ;
    l_reg1 = i_vec_reg_number_0 % 8;
    l_reg2 = i_vec_reg_number_1 % 8;
    if ( i_vec_reg_number_0 >= 8 ) { l_third  -= 0x40; }
    if ( i_vec_reg_number_1 >= 8 ) { l_second -= 0x80; }
    if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) && 
         (i_vec_reg_number_1<=15) )
    {
//     if ( i_vec_reg_number_0 >= 8 ) 
//     {
//        if ( l_bytes < 5 ) l_bytes = 5;
//     }
    } else l_bytes = 6;
 
    if ( l_bytes == 4 )
    {
       buf[i++] = 0xc5;
       buf[i++] = 0xfd - 8*l_reg1   + l_third + l_second + l_xreg + l_fpadj2;
       buf[i++] = 0x59 + l_fpadj;
       if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF ) 
       { 
          buf[i++] = 0x04; 
          l_place = i-1; 
       }
       buf[i++] = 0x00 + l_reg0    + 8*l_reg2 ;
    } else if ( l_bytes == 5 )
    {
       buf[i++] = 0xc4;
       buf[i++] = 0xc1 + l_second;
       buf[i++] = 0x7d - 8*l_reg1   + l_third + l_xreg + l_fpadj2;
       buf[i++] = 0x59 + l_fpadj;
       if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF ) 
       { 
          buf[i++] = 0x04; 
          l_place = i-1; 
       }
       buf[i++] = 0x00 + l_reg0    + 8*l_reg2 ;
    } else if ( l_bytes == 6 )
    {
       if ( i_gp_reg_base >= 8 ) { l_second -= 0x20; }

//     if ( i_vec_reg_number_0 >= 8 ) { l_third -= 0x40; }
       if ( i_vec_reg_number_0 >= 16) { l_third += 0x40; l_fourth -= 0x8; }
       if ( i_vec_reg_number_0 >= 24) { l_third -= 0x40; }

//     if ( i_vec_reg_number_1 >= 8 ) { l_second -= 0x80; }
       if ( i_vec_reg_number_1 >= 16) { l_second += 0x70; }
       if ( i_vec_reg_number_1 >= 24) { l_second -= 0x80; }

       buf[i++] = 0x62;
       buf[i++] = 0xf1 + l_second;
       buf[i++] = 0xfd - 8*l_reg1   + l_third + l_fpadj2;
       buf[i++] = 0x48 + l_fourth;
       buf[i++] = 0x59 + l_fpadj;
       if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF ) 
       { 
          buf[i++] = 0x04; 
          l_place = i-1; 
       }
       buf[i++] = 0x00 + l_reg0    + 8*l_reg2 ;
    }
    if (l_place==0) l_place = i - 1;
    if ( ((i_gp_reg_base % 8) == LIBXSMM_X86_GP_REG_RSP) && 
          (i_gp_reg_idx==LIBXSMM_X86_GP_REG_UNDEF) )
    {
       buf[i++] = 0x24;
    }
    if ( ( (i_gp_reg_base%8) == 5) && (i_displacement==0) ) 
    {
       // Registers like rbp/r13 when you have a displacement of 0, we need
       // force the single byte of zero to appear.
       l_forced_offset = 1;
    }
    i += add_offset ( l_place, i, i_displacement, l_forced_offset, l_sizereg, buf );

    io_generated_code->code_size = i;
    // *loc = i;
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
      unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
      int i = io_generated_code->code_size;
    // int i = *loc;
       unsigned int l_maxsize = io_generated_code->buffer_size;
    // unsigned int l_maxsize = 1024;
    int l_last = 0, l_first=0;
    int l_instype = 0;
    int l_place=0;
    int l_bytes=4;
    int l_forced_offset=0;
    int l_sizereg=64;

    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"Most instructions need at most 20 bytes\n");
       exit(-1);
    }
    if ( (i_gp_reg_base < LIBXSMM_X86_GP_REG_RAX) && 
         (i_gp_reg_base > LIBXSMM_X86_GP_REG_R15) && 
         (i_gp_reg_base < 0) && (i_gp_reg_base > 15) &&
         (i_gp_reg_base != LIBXSMM_X86_GP_REG_UNDEF) ) 
    {
       fprintf(stderr,"i_gp_reg_base error in libxsmm_instruction_prefetch\n");
       exit(-1);
    }
    if ( (i_gp_reg_idx  < LIBXSMM_X86_GP_REG_RAX) && 
         (i_gp_reg_idx  > LIBXSMM_X86_GP_REG_R15) && 
         (i_gp_reg_idx  < 0) && (i_gp_reg_idx  > 15) &&
         (i_gp_reg_idx  != LIBXSMM_X86_GP_REG_UNDEF) ) 
    {
       fprintf(stderr,"i_gp_reg_idx error in libxsmm_instruction_prefetch\n");
       exit(-1);
    }
    switch ( i_prefetch_instr ) {
       case LIBXSMM_X86_INSTR_PREFETCHT0:
          l_instype -= 8;
          break;
       case LIBXSMM_X86_INSTR_PREFETCHT1:
          break;
       case LIBXSMM_X86_INSTR_PREFETCHT2:
          l_instype += 8;
          break;
       case LIBXSMM_X86_INSTR_PREFETCHNTA:
          l_instype -= 16;
          break;
       case LIBXSMM_X86_INSTR_VPREFETCH0:
          fprintf(stderr,"don't yet do vprefetch0");
          exit(-1);
          break;
       case LIBXSMM_X86_INSTR_VPREFETCH1:
          fprintf(stderr,"don't yet do vprefetch1");
          exit(-1);
          break;
       default:
          fprintf(stderr,"Strange prefetch instruction");
          exit(-1);
          break;
    }
    if ( (i_gp_reg_base < 8) && (i_gp_reg_idx==LIBXSMM_X86_GP_REG_UNDEF) )
    {
       // prefetcht1 (%rax)
       buf[i++] = 0x0f;
       buf[i++] = 0x18;
       buf[i++] = 0x10 + l_instype + i_gp_reg_base;
       l_place = i-1;
       if ( i_gp_reg_base == LIBXSMM_X86_GP_REG_RSP ) 
       {
          buf[i++] = 0x24;
       }
    } else if (i_gp_reg_idx==LIBXSMM_X86_GP_REG_UNDEF) {
       // prefetcht1 (%r8)
       buf[i++] = 0x41;
       buf[i++] = 0x0f;
       buf[i++] = 0x18;
       buf[i++] = 0x10 + l_instype + i_gp_reg_base - 8;
       l_place = i-1;
       if ( i_gp_reg_base == LIBXSMM_X86_GP_REG_R12 ) 
       {
          buf[i++] = 0x24;
       }
    } else { // two GP regs are being used
       l_last = 0;
       if ( i_scale == 2 ) l_last = 0x40;
       else if ( i_scale == 4 ) l_last = 0x80;
       else if ( i_scale == 8 ) l_last = 0xc0;
       l_bytes = 4;
       l_first = 0;
       if ( i_gp_reg_base < 8 )
       {
          l_last += i_gp_reg_base;
          if ( i_gp_reg_idx >= 8 ) 
          { 
             l_first = 1; 
             l_bytes = 5; 
             l_last += 8*(i_gp_reg_idx - 8);
          } else {
             l_last += 8*(i_gp_reg_idx);
          }
       } else {
          l_last += i_gp_reg_base-8;
          l_bytes = 5;
          if ( i_gp_reg_idx >= 8 )
          {
             l_first = 2;
             l_last += 8*(i_gp_reg_idx - 8);
          } else {
             l_last += 8*(i_gp_reg_idx);
          }
       }
       if ( l_bytes == 5 )
       {
          buf[i++] = 0x41 + l_first;
       }
       buf[i++] = 0x0f;
       buf[i++] = 0x18;
       buf[i++] = 0x14 + l_instype;
       l_place = i - 1;
       buf[i++] = 0x00 + l_last;
    }
    l_sizereg = 1;
    if ( ( (i_gp_reg_base%8) == 5) && (i_displacement==0) )
    {
       // Registers like rbp/r13 when you have a displacement of 0, we need
       // force the single byte of zero to appear.
       l_forced_offset = 1;
    }
    i += add_offset ( l_place, i, i_displacement, l_forced_offset, l_sizereg, buf );

    io_generated_code->code_size = i;
    // *loc = i;
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


    /* this is a very simple System V ABI 64 interfacce */
    unsigned char* l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (l_max_size < (l_code_size + 9)) {
      fprintf(stderr, "Jit buffer to small\n!");
      exit(-1);
    }

    /* push rbx */
    l_code_buffer[l_code_size++] = 0x53;
    /* push r12 */
    l_code_buffer[l_code_size++] = 0x41;
    l_code_buffer[l_code_size++] = 0x54;
    /* push r13 */
    l_code_buffer[l_code_size++] = 0x41;
    l_code_buffer[l_code_size++] = 0x55;
    /* push r14 */
    l_code_buffer[l_code_size++] = 0x41;
    l_code_buffer[l_code_size++] = 0x56;
    /* push r15 */
    l_code_buffer[l_code_size++] = 0x41;
    l_code_buffer[l_code_size++] = 0x57;

    /* update code length */
    io_generated_code->code_size = l_code_size;
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

    /* this is a very simple System V ABI 64 interfacce */
    unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (l_max_size < (l_code_size + 10)) {
      fprintf(stderr, "Jit buffer to small\n!" );
      exit(-1);
    }

    /* pop r15 */
    l_code_buffer[l_code_size++] = 0x41;
    l_code_buffer[l_code_size++] = 0x5f;
    /* pop r14 */
    l_code_buffer[l_code_size++] = 0x41;
    l_code_buffer[l_code_size++] = 0x5e;
    /* pop r13 */
    l_code_buffer[l_code_size++] = 0x41;
    l_code_buffer[l_code_size++] = 0x5d;
    /* pop r12 */
    l_code_buffer[l_code_size++] = 0x41;
    l_code_buffer[l_code_size++] = 0x5c;
    /* pop rbx */
    l_code_buffer[l_code_size++] = 0x5b;
    /* retq */
    l_code_buffer[l_code_size++] = 0xc3;

    /* update code length */
    io_generated_code->code_size = l_code_size;
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

