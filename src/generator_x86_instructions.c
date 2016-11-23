/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.), Greg Henry (Intel Corp.)
******************************************************************************/
#include "generator_x86_instructions.h"
#include "generator_common.h"

#include <libxsmm_cpuid.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* This routine is for the jit code. All offsets/displacements have similar
   byte patterns, so this is used for all of them */
LIBXSMM_INLINE
int add_offset (const unsigned int i_place1,
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
      buf[i_place2] = (unsigned char)(i_offset/i_sizereg);
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

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_vec_move( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_instruction_set,
                                       const unsigned int      i_vmove_instr,
                                       const unsigned int      i_gp_reg_base,
                                       const unsigned int      i_gp_reg_idx,
                                       const unsigned int      i_scale,
                                       const int               i_displacement,
                                       const char              i_vector_name,
                                       const unsigned int      i_vec_reg_number_0,
                                       const unsigned int      i_use_masking,
                                       const unsigned int      i_is_store )
{
/* Greg asks: do we still need this condition? It seems to me this works now
#ifndef NDEBUG
  if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_NO_INDEX_SCALE_ADDR );
    return;
  }
#endif
*/
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /* unsigned int l_maxsize = 1024; */
    int l_iregnum = i_gp_reg_base   % 8;
    int l_vregnum = i_vec_reg_number_0 % 8;
    int l_ivectype=0, l_ivectype2=0, l_iregoff=0, l_ivectype3=0;
    int l_vregoffset=0, l_vregoffset2=0;
    int l_aligned=0, l_forced_offset=0, l_penultimate=0;
    int l_place, l_num=0, l_num2=0, l_num3=0, l_sizereg=1;
    int l_maskingoff=0;
    int l_wow = 0;
    int l_tscale= 0;
    int l_bytes = 4; /* base number of bytes */

    int i_mask_reg_number = 1; /* change if you don't want k1 */

    if ( (i_vector_name != 'z') && (i_use_masking!=0) )
    {
       fprintf(stderr,"libxsmm_instruction_vec_move: Masking is only enabled with zmm registers!\n");
       exit(-1);
    }
    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"libxsmm_instruction_vec_move: Most instructions need at most 20 bytes\n");
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
          if ( i_vector_name!='x' ) l_ivectype -= 1; /* single */
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVSS:
          if ( i_vector_name!='x' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: You want to use vmovss without xmm?\n");
             exit(-1);
          }
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_ivectype += 2;
          break;
       case LIBXSMM_X86_INSTR_VMOVSD:
          if ( i_vector_name!='x' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: You want to use vmovsd without xmm?\n");
             exit(-1);
          }
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_ivectype += 3;
          break;
       case LIBXSMM_X86_INSTR_VPBROADCASTD:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vpbroadcastd not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vpbroadcastd and store?\n");
             exit(-1);
          }
          l_ivectype2 += 0x01;
          l_penultimate += 0x48;
          l_num2 += 1;
          l_num3 += 0x21;
          l_sizereg = 4;
          break;
       case LIBXSMM_X86_INSTR_VPBROADCASTQ:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vpbroadcastq not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vpbroadcastq and store?\n");
             exit(-1);
          }
          l_ivectype2 += 0x81;
          l_penultimate += 0x49;
          l_num2 += 1;
          l_num3 += 0x21;
          l_sizereg = 8;
          break;
       case LIBXSMM_X86_INSTR_VPBROADCASTB:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vpbroadcastb not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vpbroadcastb and store?\n");
             exit(-1);
          }
          l_ivectype2 += 0x01;
          l_penultimate += 0x68;
          l_num2 += 1;
          l_num3 += 0x21;
          l_sizereg = 1;
          break;
       case LIBXSMM_X86_INSTR_VPBROADCASTW:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vpbroadcastw not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vpbroadcastw and store?\n");
             exit(-1);
          }
          l_ivectype2 += 0x01;
          l_penultimate += 0x69;
          l_num2 += 1;
          l_num3 += 0x21;
          l_sizereg = 2;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQA32:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vmovdqa32 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x01;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQA64:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vmovdqa64 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x81;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQU8:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vmovdqu8 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x03;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQU16:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vmovdqu16 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x83;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQU32:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vmovdqu32 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x02;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQU64:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vmovdqu64 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x82;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VBROADCASTSD:
          l_bytes = 5;
          if ( i_vector_name=='x' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vbroadcastsd and xmm?\n");
             exit(-1);
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vbroadcastsd and stores?\n");
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
             l_ivectype += 1;
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vbroadcastss and stores?\n");
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
          if ( i_vector_name!='x' ) l_ivectype -= 1; /* single */
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVDDUP:
          if ( i_is_store == 1 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vmovddup and stores?\n");
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
          fprintf(stderr,"libxsmm_instruction_vec_move: unexpected instruction number: %u\n",i_vmove_instr);
          exit(-1);
    }
    switch ( i_vector_name ) {
       case 'x':
          l_sizereg = 1;
          if ( l_num > 1 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: Are you sure xmm%u exists?\n",i_vec_reg_number_0);
             exit(-1);
          }
          break;
       case 'y':
          l_ivectype += 5;
          l_sizereg = 1;
          if ( l_num > 2 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: Are you sure ymm%u exists?\n",i_vec_reg_number_0);
             exit(-1);
          }
          break;
       case 'z':
          l_bytes = 6;
          break;
       default:
          fprintf(stderr,"libxsmm_instruction_vec_move: Exactly what sort of fp regs are you using?\n");
          exit(-1);
    }
    if ( (i_gp_reg_base >= 8) && (i_gp_reg_base <=15) )
    {
       if ( l_bytes < 5 ) l_bytes = 5;
       else l_iregoff -= 0x20;
    }
    if ( (i_gp_reg_idx>=8) && (i_gp_reg_idx<=15) )
    {
       if ( l_bytes < 5 )
       {
          l_bytes = 5;
       } else {
          l_wow -= 0x20;
       }
       l_wow -= 0x20;
    }

    if ( i_is_store == 1 )
    {
       l_aligned += 1;
       /*if ( i_use_masking != 0 ) l_maskingoff = i_mask_reg_number;*/
    } else {
       /*The following addition of 0x80 appears broken...
        if ( i_use_masking != 0 ) l_maskingoff = 0x80 + i_mask_reg_number; */
    }
    if ( i_use_masking != 0 ) l_maskingoff = i_mask_reg_number;

    if ( l_num == 0 ) l_vregoffset = 0x90;
    else if ( l_num == 1 ) { l_vregoffset = 0x10; l_vregoffset2 = -0x80; }
    else if ( l_num == 2 ) l_vregoffset = 0x80;
    else if ( l_num == 3 ) l_vregoffset = 0x00;
    if ( (l_iregnum == 5) && (i_displacement==0) )
    {
       /* Registers like rbp/r13 when you have a displacement of 0, we need
          force the single byte of zero to appear. */
       l_forced_offset=1;
    }

    if ( l_bytes == 4 )
    {
       buf[i++] = 0xc5;
       buf[i++] = (unsigned char)(0xf8 + l_ivectype + l_ivectype3);
    } else if ( l_bytes == 5 ) {
       buf[i++] = 0xc4;
       buf[i++] = (unsigned char)(0xc1 + l_num3 + l_vregoffset2 + l_iregoff + l_wow);
       buf[i++] = (unsigned char)(0x78 + l_ivectype);
    } else if ( l_bytes == 6 ) {
       buf[i++] = 0x62;
       buf[i++] = (unsigned char)(0x61 + l_vregoffset + l_iregoff + l_num2 + l_wow);
       buf[i++] = (unsigned char)(0x7c + l_ivectype2);
       buf[i++] = (unsigned char)(0x48 + l_maskingoff);
    }
    buf[i++] = (unsigned char)(0x10 + l_aligned + l_penultimate);
    if ( (i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF) &&
    ((int)i_gp_reg_idx >= LIBXSMM_X86_GP_REG_RAX) &&
         (i_gp_reg_idx <= LIBXSMM_X86_GP_REG_R15) )
    {
       buf[i++] = (unsigned char)(0x04 + 8*l_vregnum);
       l_place = i-1;
       if ( i_scale == 1 ) l_tscale = 0x00;
       else if ( i_scale == 2 ) l_tscale = 0x40;
       else if ( i_scale == 4 ) l_tscale = 0x80;
       else if ( i_scale == 8 ) l_tscale = 0xc0;
       else
       {
          fprintf(stderr,"libxsmm_instruction_vec_move: Don't understand the i_scale parameter");
          exit(-1);
       }
       buf[i++] = (unsigned char)(l_tscale + l_iregnum + 8*(i_gp_reg_idx%8));
    } else {
       l_place = i;
       buf[i++] = (unsigned char)(0x00 + l_iregnum + 8*l_vregnum);
       if ( l_iregnum == LIBXSMM_X86_GP_REG_RSP )
       {
          buf[i++] = 0x24;
       }
    }
    i += add_offset ( l_place, i, i_displacement, l_forced_offset, l_sizereg, buf );

    io_generated_code->code_size = i;
    /* *loc = i; */
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_base_name[4];
    char l_instr_name[16];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base_name, 3 );
    libxsmm_get_x86_instr_name( i_vmove_instr, l_instr_name, 15 );

    if ( (i_instruction_set == LIBXSMM_X86_AVX512_MIC   ||
          i_instruction_set == LIBXSMM_X86_AVX512_CORE     ) &&
         (i_use_masking != 0) ) {
      /* build vmovpd/ps/sd/ss instruction, load use */
      if ( i_is_store == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s), %%%%%cmm%u%%{%%%%k%i%%}%%{z%%}\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%%cmm%u{%%k%i}{z}\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %i(%%%%%s)%%{%%%%k%i%%}\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %i(%%%s){%%k%i}\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      }
    } else if ( (i_instruction_set == LIBXSMM_X86_IMCI) && (i_use_masking != 0) ) {
      /* build vmovpd/ps/sd/ss instruction, load use */
      if ( i_is_store == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s), %%%%%cmm%u%%{%%%%k%i%%}\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%%cmm%u{%%k%i}\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %i(%%%%%s)%%{%%%%k%i%%}\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %i(%%%s){%%k%i}\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name, LIBXSMM_X86_IMCI_AVX512_MASK );
        }
      }
    } else {
      /* build vmovpd/ps/sd/ss instruction, load use */
      if ( i_is_store == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s), %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0 );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0 );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %i(%%%%%s)\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %i(%%%s)\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name );
        }
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_vec_compute_reg( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_instruction_set,
                                              const unsigned int      i_vec_instr,
                                              const char              i_vector_name,
                                              const unsigned int      i_vec_reg_number_0,
                                              const unsigned int      i_vec_reg_number_1,
                                              const unsigned int      i_vec_reg_number_2 )
{
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /* unsigned int l_maxsize = 1024; */
    int l_second=0, l_third=0, l_fourth=0, l_xreg=0;
    int l_reg0   = i_vec_reg_number_0;
    int l_reg1   = i_vec_reg_number_1;
    int l_reg2   = i_vec_reg_number_2;
    int l_fpadj=0;
    int l_fpadj2=0;
    int l_bytes=4;

    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"libxsmm_instruction_vec_compute_reg: Most instructions need at most 20 bytes\n");
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
       case LIBXSMM_X86_INSTR_VPADDD:
          l_fpadj2 -= 0x80;
          l_fpadj  += 0xA5;
          break;
       case LIBXSMM_X86_INSTR_VPADDQ:
          l_fpadj  += 0x7b;
          break;
       case LIBXSMM_X86_INSTR_VPADDW:
          l_fpadj2 -= 0x80;
          l_fpadj  += 0xA4;
          break;
       case LIBXSMM_X86_INSTR_VPADDB:
          l_fpadj2 -= 0x80;
          l_fpadj  += 0xA3;
          break;
       case LIBXSMM_X86_INSTR_VPMADDWD:
          l_fpadj2 -= 0x80;
          l_fpadj  += 0x9C;
          break;
       case LIBXSMM_X86_INSTR_VPMADDUBSW:
          l_second += 0x01;
          l_fpadj  -= 0x55;
          l_fpadj2 -= 0x80;
          break;
       case LIBXSMM_X86_INSTR_VPADDSW:
          l_fpadj  += 0x94;
          l_fpadj2 -= 0x80;
          break;
       case LIBXSMM_X86_INSTR_VPADDSB:
          l_fpadj  += 0x93;
          l_fpadj2 -= 0x80;
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
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VMULSD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VADDSD:
          l_fpadj  =-1;
          l_fpadj2 = 2;
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VADDSD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VSUBSD:
          l_fpadj  = 3;
          l_fpadj2 = 2;
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VSUBSD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SD:
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: Really? VFMADD231SD and ymm/zmm?\n");
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
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VFMSUB231SD and ymm/zmm?\n");
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
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VFNMADD231SD and ymm/zmm?\n");
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
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VFNMSUB231SD and ymm/zmm?\n");
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
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VMULSS and ymm/zmm?\n");
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VADDSS:
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VADDSS and ymm/zmm?\n");
          l_fpadj  =-1;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VSUBSS:
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VSUBSS and ymm/zmm?\n");
          l_fpadj  = 3;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SS:
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VFMADD231SS and ymm/zmm?\n");
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
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VFMSUB231SS and ymm/zmm?\n");
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
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VFNMADD231SS and ymm/zmm?\n");
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
          if (i_vector_name != 'x') fprintf(stderr,"libxsmm_instruction_vec_compute_reg: VFNMSUB231SS and ymm/zmm?\n");
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
          fprintf(stderr, "libxsmm_instruction_vec_compute_reg: Unknown instruction type: %u\n", i_vec_instr);
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
       buf[i++] = (unsigned char)(0xfd - 8*l_reg1   + l_third + l_second + l_xreg + l_fpadj2);
       buf[i++] = (unsigned char)(0x59 + l_fpadj);
       buf[i++] = (unsigned char)(0xc0 + l_reg0    + 8*l_reg2);
    } else if ( l_bytes == 5 )
    {
       buf[i++] = 0xc4;
       buf[i++] = (unsigned char)(0xc1 + l_second);
       buf[i++] = (unsigned char)(0x7d - 8*l_reg1   + l_third + l_xreg + l_fpadj2);
       buf[i++] = (unsigned char)(0x59 + l_fpadj);
       buf[i++] = (unsigned char)(0xc0 + l_reg0    + 8*l_reg2);
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
       buf[i++] = (unsigned char)(0xf1 + l_second);
       buf[i++] = (unsigned char)(0xfd - 8*l_reg1   + l_third + l_fpadj2);
       buf[i++] = (unsigned char)(0x48 + l_fourth);
       buf[i++] = (unsigned char)(0x59 + l_fpadj);
       buf[i++] = (unsigned char)(0xc0 + l_reg0    + 8*l_reg2);
    }

    io_generated_code->code_size = i;
    /* *loc = i; */

  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name, 15 );

    /* build vXYZpd/ps/sd/ss instruction pure register use*/
    if ( i_instruction_set != LIBXSMM_X86_SSE3 ) {
      if ( io_generated_code->code_type == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2 );
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2 );
      }
    } else {
      if ( io_generated_code->code_type == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1);
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %%%cmm%u\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_vec_compute_mem( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_instruction_set,
                                              const unsigned int      i_vec_instr,
                                              const unsigned int      i_use_broadcast,
                                              const unsigned int      i_gp_reg_base,
                                              const unsigned int      i_gp_reg_idx,
                                              const unsigned int      i_scale,
                                              const int               i_displacement,
                                              const char              i_vector_name,
                                              const unsigned int      i_vec_reg_number_0,
                                              const unsigned int      i_vec_reg_number_1 ) {
  /* @TODO add checks in debug mode */
  if ( (i_instruction_set != LIBXSMM_X86_IMCI)        &&
       (i_instruction_set != LIBXSMM_X86_AVX512_MIC)  &&
       (i_instruction_set != LIBXSMM_X86_AVX512_CORE) &&
       (i_use_broadcast != 0)                            ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_NO_IMCI_AVX512_BCAST );
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /*int i = *loc;*/
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /*unsigned int l_maxsize = 1024;*/
    int l_second=0, l_third=0, l_fourth=0, l_xreg=0;
    int l_reg0 = 0;
    int l_reg1   = i_vec_reg_number_0;
    int l_reg2   = i_vec_reg_number_1;
    int l_fpadj=0, l_place=0;
    int l_fpadj2=0;
    int l_bytes=4;
    int l_regi=0;
    int l_forced_offset=0;
    int l_sizereg=64;
    int l_scaleadj=0;
    /* int l_iregoff = 0; */

    int l_broadcast = (int)i_use_broadcast;

    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"libxsmm_instruction_vec_compute_mem: Most instructions need at most 20 bytes\n");
       exit(-1);
    }
    switch ( i_vector_name ) {
       case 'x':
          l_sizereg = 1;
          if ( l_broadcast != 0 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: broadcasts aren't enabled with xmm yet\n");
             exit(-1);
          }
          break;
       case 'y':
          l_sizereg = 1;
          if ( l_broadcast != 0 )
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: broadcasts aren't enabled with ymm yet\n");
             exit(-1);
          }
          break;
       case 'z':
          l_bytes = 6;
          break;
       default:
          fprintf(stderr,"libxsmm_instruction_vec_compute_mem: Unknown sort of fp registers\n");
          exit(-1);
    }
    if ( l_broadcast != 0 ) l_sizereg = 8;
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
          } else if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
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
          } else if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
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
          } else if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
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
          } else if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VMULSD:
          l_fpadj2 = 2;
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vmulsd and ymm/zmm?\n");
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_VADDSD:
          l_fpadj  =-1;
          l_fpadj2 = 2;
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vaddsd and ymm/zmm?\n");
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_VSUBSD:
          l_fpadj  = 3;
          l_fpadj2 = 2;
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vsubsd and ymm/zmm?\n");
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SD:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vfmadd231sd and ymm/zmm?\n");
             exit(-1);
          }
          l_second += 0x21;
          l_fpadj  += 0x60;
          l_fpadj2 += 0x80;
          if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231SD:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vfmsub231sd and ymm/zmm?\n");
             exit(-1);
          }
          l_second += 0x21;
          l_fpadj  += 0x62;
          l_fpadj2 += 0x80;
          if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231SD:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vfnmadd231sd and ymm/zmm?\n");
             exit(-1);
          }
          l_second += 0x21;
          l_fpadj  += 0x64;
          l_fpadj2 += 0x80;
          if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231SD:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vfnmsub231sd and ymm/zmm?\n");
             exit(-1);
          }
          l_second += 0x21;
          l_fpadj  += 0x66;
          l_fpadj2 += 0x80;
          if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VXORPS:
          l_fpadj2 = -1;
          l_fpadj = -2;
          if ( l_broadcast == 1 ) l_sizereg = 4;
          if ( i_vector_name == 'z' )
          {
             l_fpadj2 -= 0x80;
          }
          break;
       case LIBXSMM_X86_INSTR_VMULPS:
          if ( l_broadcast == 1 ) l_sizereg = 4;
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) &&
               (i_vec_reg_number_1<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          break;
       case LIBXSMM_X86_INSTR_VADDPS:
          if ( l_broadcast == 1 ) l_sizereg = 4;
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) &&
               (i_vec_reg_number_1<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = -1;
          break;
       case LIBXSMM_X86_INSTR_VSUBPS:
          if ( l_broadcast == 1 ) l_sizereg = 4;
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) &&
               (i_vec_reg_number_1<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = 3;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231PS:
          if ( l_broadcast == 1 ) l_sizereg = 4;
          l_second += 0x21;
          l_fpadj  += 0x5f;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231PS:
          if ( l_broadcast == 1 ) l_sizereg = 4;
          l_second += 0x21;
          l_fpadj  += 0x61;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231PS:
          if ( l_broadcast == 1 ) l_sizereg = 4;
          l_second += 0x21;
          l_fpadj  += 0x63;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231PS:
          if ( l_broadcast == 1 ) l_sizereg = 4;
          l_second += 0x21;
          l_fpadj  += 0x65;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VMULSS:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vmulss and ymm/zmm?\n");
             exit(-1);
          }
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VADDSS:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vaddss and ymm/zmm?\n");
             exit(-1);
          }
          l_fpadj  =-1;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VSUBSS:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vsubss and ymm/zmm?\n");
             exit(-1);
          }
          l_fpadj  = 3;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SS:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vfmadd231ss and ymm/zmm?\n");
             exit(-1);
          }
          l_second += 0x21;
          l_fpadj  += 0x60;
          if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231SS:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vfmsub231ss and ymm/zmm?\n");
             exit(-1);
          }
          l_second += 0x21;
          l_fpadj  += 0x62;
          if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231SS:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vfnmadd231ss and ymm/zmm?\n");
             exit(-1);
          }
          l_second += 0x21;
          l_fpadj  += 0x64;
          if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231SS:
          if (i_vector_name != 'x')
          {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem: vfnmsub231ss and ymm/zmm?\n");
             exit(-1);
          }
          l_second += 0x21;
          l_fpadj  += 0x66;
          if ( (i_gp_reg_base > 7) && (i_gp_reg_base <= 15 ) ) {
             l_second -= 0x20;
          } else if ( (i_gp_reg_idx > 7) && (i_gp_reg_idx<=15) ) {
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
          if ( l_broadcast != 0 ) l_sizereg = 4;
          l_fpadj += 0x96;
          l_fpadj2 += 0x80;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_mem: Unknown instruction type: %u\n", i_vec_instr);
          break;
    }
    if ( (i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF) &&
    ((int)i_gp_reg_idx >= LIBXSMM_X86_GP_REG_RAX) &&
         (i_gp_reg_idx <= LIBXSMM_X86_GP_REG_R15) )
    {
       switch ( i_scale ) {
          case 1:
             l_scaleadj=0;
             break;
          case 2:
             l_scaleadj=0x40;
             break;
          case 4:
             l_scaleadj=0x80;
             break;
          case 8:
             l_scaleadj=0xc0;
             break;
          default:
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: i_scale parameter appears bogus=%u\n", i_scale);
             exit(-1);
             break;
       }
    }
    if ( (i_gp_reg_base >= 8) && (i_gp_reg_base != LIBXSMM_X86_GP_REG_UNDEF) )
    {
       if ( l_bytes < 5 ) l_bytes = 5;
       /* else l_iregoff -= 0x20; */
    }
    l_regi = i_gp_reg_idx;
    if ( (i_gp_reg_idx  >= 8) && (i_gp_reg_idx  != LIBXSMM_X86_GP_REG_UNDEF) )
    {
       if ( l_bytes < 5 ) l_bytes = 5;
       l_regi = i_gp_reg_idx-8;
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
#if 0
     if ( i_vec_reg_number_0 >= 8 )
     {
        if ( l_bytes < 5 ) l_bytes = 5;
     }
#endif
    } else l_bytes = 6;


    if ( l_bytes == 4 )
    {
       buf[i++] = 0xc5;
       buf[i++] = (unsigned char)(0xfd - 8*l_reg1   + l_third + l_second + l_xreg + l_fpadj2);
       buf[i++] = (unsigned char)(0x59 + l_fpadj);
       if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF )
       {
          buf[i++] = (unsigned char)(0x04 + 8*l_reg2);
          l_place = i-1;
          buf[i++] = (unsigned char)(0x00 + l_reg0 + l_scaleadj + 8*l_regi);
       } else {
          buf[i++] = (unsigned char)(0x00 + l_reg0 + 8*l_reg2);
       }
    } else if ( l_bytes == 5 )
    {
       if ((i_gp_reg_base >= 8) && (i_gp_reg_base != LIBXSMM_X86_GP_REG_UNDEF))
       {
          if ((i_gp_reg_idx >= 8) && (i_gp_reg_idx  != LIBXSMM_X86_GP_REG_UNDEF))
          {
             l_second -= 0x20;
          }
       }
       if ((i_gp_reg_idx >= 8) && (i_gp_reg_idx  != LIBXSMM_X86_GP_REG_UNDEF))
       {
          l_second -= 0x20;
       }
       buf[i++] = 0xc4;
       buf[i++] = (unsigned char)(0xc1 + l_second);
       buf[i++] = (unsigned char)(0x7d - 8*l_reg1   + l_third + l_xreg + l_fpadj2);
       buf[i++] = (unsigned char)(0x59 + l_fpadj);
       if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF )
       {
          buf[i++] = (unsigned char)(0x04 + 8*l_reg2);
          l_place = i-1;
          buf[i++] = (unsigned char)(0x00 + l_reg0 + l_scaleadj + 8*l_regi);
       } else {
          buf[i++] = (unsigned char)(0x00 + l_reg0 + 8*l_reg2);
       }
    } else if ( l_bytes == 6 )
    {
       if ( i_gp_reg_base >= 8 ) { l_second -= 0x20; }
       if ( (i_gp_reg_idx >= 8) && (i_gp_reg_idx  != LIBXSMM_X86_GP_REG_UNDEF) )
       {
          l_second -= 0x40;
       }

/*     if ( i_vec_reg_number_0 >= 8 ) { l_third -= 0x40; } */
       if ( i_vec_reg_number_0 >= 16) { l_third += 0x40; l_fourth -= 0x8; }
       if ( i_vec_reg_number_0 >= 24) { l_third -= 0x40; }

/*     if ( i_vec_reg_number_1 >= 8 ) { l_second -= 0x80; } */
       if ( i_vec_reg_number_1 >= 16) { l_second += 0x70; }
       if ( i_vec_reg_number_1 >= 24) { l_second -= 0x80; }
       if ( l_broadcast != 0 ) { l_fourth += 0x10; }

       buf[i++] = 0x62;
       buf[i++] = (unsigned char)(0xf1 + l_second);
       buf[i++] = (unsigned char)(0xfd - 8*l_reg1   + l_third + l_fpadj2);
       buf[i++] = (unsigned char)(0x48 + l_fourth);
       buf[i++] = (unsigned char)(0x59 + l_fpadj);
       if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF )
       {
          buf[i++] = (unsigned char)(0x04 + 8*l_reg2);
          l_place = i-1;
          buf[i++] = (unsigned char)(0x00 + l_reg0 + l_scaleadj + 8*l_regi);
       } else {
          buf[i++] = (unsigned char)(0x00 + l_reg0    + 8*l_reg2);
       }
    }
    if (l_place==0) l_place = i - 1;
    if ( ((i_gp_reg_base % 8) == LIBXSMM_X86_GP_REG_RSP) &&
          (i_gp_reg_idx==LIBXSMM_X86_GP_REG_UNDEF) )
    {
       buf[i++] = 0x24;
    }
    if ( ( (i_gp_reg_base%8) == 5) && (i_displacement==0) )
    {
       /* Registers like rbp/r13 when you have a displacement of 0, we need
          force the single byte of zero to appear. */
       l_forced_offset = 1;
    }
    i += add_offset ( l_place, i, i_displacement, l_forced_offset, l_sizereg, buf );

    io_generated_code->code_size = i;
    /* *loc = i; */

  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_base[4];
    char l_gp_reg_idx[4];
    char l_instr_name[16];
    char l_broadcast[8];
    unsigned int l_single_precision = libxsmm_is_x86_vec_instr_single_precision( i_vec_instr );

    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base, 3 );
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name, 15 );

    if (l_single_precision == 0) {
      LIBXSMM_SNPRINTF( l_broadcast, 7, "1to8" );
    } else {
      LIBXSMM_SNPRINTF( l_broadcast, 7, "1to16" );
    }

    /* build vXYZpd/ps/sd/ss instruction pure register use*/
    if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF ) {
      if ( io_generated_code->code_type == 0 ) {
        if (i_use_broadcast != 0) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s)%%{%s%%}, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s), %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        }
      } else {
        if (i_use_broadcast != 0) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s){%s}, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        }
      }
    } else {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx, 3 );
      if ( io_generated_code->code_type == 0 ) {
        if (i_use_broadcast != 0) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%%s,%u)%%{%s%%}, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%%s,%u), %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        }
      } else {
        if (i_use_broadcast != 0) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u){%s}, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u), %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        }
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_vec_shuffle_reg( libxsmm_generated_code* io_generated_code,
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
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name, 15 );

    if ( i_instruction_set != LIBXSMM_X86_SSE3 ) {
      if ( io_generated_code->code_type == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s $%u, %%%%%cmm%u, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_shuffle_operand, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2 );
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%u, %%%cmm%u, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_shuffle_operand, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2 );
      }
    } else {
      if ( io_generated_code->code_type == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s $%u, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_shuffle_operand, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%u, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_shuffle_operand, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_vec_move_gathscat( libxsmm_generated_code* io_generated_code,
                                                const unsigned int      i_instruction_set,
                                                const unsigned int      i_vmove_instr,
                                                const char              i_vector_name,
                                                const unsigned int      i_gp_reg_base,
                                                const unsigned int      i_vec_reg_idx,
                                                const unsigned int      i_scale,
                                                const int               i_displacement,
                                                const unsigned int      i_vec_reg_number,
                                                const unsigned int      i_mask_reg_number,
                                                const unsigned int      i_is_gather ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /* unsigned int l_maxsize = 1024; */
    int l_sizereg = 0;
    int l_instr_offset = 0;
    int l_instr_offset2 = 0;

    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"libxsmm_x86_instruction_vec_move_gathscat: Most instructions need at most 20 bytes\n");
       exit(-1);
    }
    switch ( i_vmove_instr ) {
       case LIBXSMM_X86_INSTR_VGATHERDPS:
          l_sizereg = 4;
          l_instr_offset = 0;
          l_instr_offset2 = 0;
          break;
       case LIBXSMM_X86_INSTR_VGATHERDPD:
          l_sizereg = 8;
          l_instr_offset = 0x80;
          l_instr_offset2 = 0;
          break;
       case LIBXSMM_X86_INSTR_VGATHERQPS:
          l_sizereg = 4;
          l_instr_offset = 0;
          l_instr_offset2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VGATHERQPD:
          l_sizereg = 8;
          l_instr_offset = 0x80;
          l_instr_offset2 = 1;
          break;
       default:
          fprintf(stderr,"libxsmm_x86_instruction_vec_move_gathscat: Strange gather/scatter instruction");
          exit(-1);
          break;
    }
    if ( i_vector_name != 'z' )
    {
       fprintf(stderr,"libxsmm_x86_instruction_vec_move_gathscat: encoder only implemented for zmm registers, but notice that i_vector_name=%c\n",i_vector_name);
       exit(-1);
    }
    if ( i_is_gather == 0 )
    {
       fprintf(stderr,"libxsmm_x86_instruction_vec_move_gathscat: encoder not implemented for scatters yet\n");
       exit(-1);
    }

    { /* open a new scope to avoid warning about mixed declaration and code (C89) */
      int l_regbas0 = i_gp_reg_base % 8;
      int l_gp8     = ((i_gp_reg_base > 7)&&(i_gp_reg_base<=15)?1:0);
      int l_vecval0 = i_vec_reg_number % 8;
      int l_vecgrp0 = i_vec_reg_number / 8;
      int l_oddgrp0 = ((l_vecgrp0 % 2)==1);
      int l_2or3grp0 = (l_vecgrp0>=2);
      int l_vecval1 = i_vec_reg_idx % 8;
      int l_vecgrp1 = i_vec_reg_idx / 8;
      int l_oddgrp1 = ((l_vecgrp1 % 2)==1);
      int l_2or3grp1 = (l_vecgrp1>=2);
      int l_sca=0;

      if (i_scale==2) l_sca=0x40;
      else if (i_scale==4) l_sca=0x80;
      else if (i_scale==8) l_sca=0xc0;

      buf[i++] = 0x62;
      buf[i++] = (unsigned char)(0xf2 - l_gp8 * 0x20 - l_oddgrp0 * 0x40 - l_oddgrp1 * 0x80 - l_2or3grp1 * 0x10);
      buf[i++] = (unsigned char)(0x7d + l_instr_offset);
      buf[i++] = (unsigned char)(0x48 - l_2or3grp0 * 0x08 + i_mask_reg_number);
      buf[i++] = (unsigned char)(0x92 + l_instr_offset2);
      buf[i++] = (unsigned char)(0x04 + l_vecval1 * 8);
      buf[i++] = (unsigned char)(0x00 + l_sca + l_regbas0 + l_vecval0 * 8);
      i += add_offset ( 5, 7, i_displacement, 0, l_sizereg, buf );

      io_generated_code->code_size = i;
      /* *loc = i; */
    }

  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[16];
    char l_gp_reg_base_name[4];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base_name, 3 );
    libxsmm_get_x86_instr_name( i_vmove_instr, l_instr_name, 15 );

    if ( i_is_gather == 0 ) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_x86_instruction_vec_move_gathscat yet needs to be implemented for scatters!\n");
      exit(-1);
    } else {
      if ( i_instruction_set == LIBXSMM_X86_AVX512_MIC || i_instruction_set == LIBXSMM_X86_AVX512_CORE ) {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%zmm%u,%u), %%%%zmm%u%%{%%%%k%u%%}\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vec_reg_idx, i_scale, i_vec_reg_number, i_mask_reg_number);
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%zmm%u,%u), %%zmm%u{%%k%u}\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vec_reg_idx, i_scale, i_vec_reg_number, i_mask_reg_number );
      }
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      } else {
        fprintf(stderr, "LIBXSMM ERROR: libxsmm_x86_instruction_vec_move_gathscat yet needs to be implemented for non-AVX512F!\n");
        exit(-1);
      }
    }
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_prefetch( libxsmm_generated_code* io_generated_code,
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
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /* unsigned int l_maxsize = 1024; */
    int l_last = 0, l_first=0;
    int l_instype = 0;
    int l_place=0;
    int l_bytes=4;
    int l_forced_offset=0;
    int l_sizereg=64;

    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"libxsmm_instruction_prefetch: Most instructions need at most 20 bytes\n");
       exit(-1);
    }
    if ( ((int)i_gp_reg_base < LIBXSMM_X86_GP_REG_RAX) ||
         ((int)i_gp_reg_base > LIBXSMM_X86_GP_REG_R15) ||
         (i_gp_reg_base > 15) ||
         ((int)i_gp_reg_base == LIBXSMM_X86_GP_REG_UNDEF) )
    {
       fprintf(stderr,"libxsmm_instruction_prefetch: i_gp_reg_base error in libxsmm_instruction_prefetch\n");
       exit(-1);
    }
    if ( (int)i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF )
    {
       fprintf(stderr,"libxsmm_instruction_prefetch: i_gp_reg_idx error in libxsmm_instruction_prefetch\n");
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
          fprintf(stderr,"libxsmm_instruction_prefetch: don't yet do vprefetch0\n");
          exit(-1);
          break;
       case LIBXSMM_X86_INSTR_VPREFETCH1:
          fprintf(stderr,"libxsmm_instruction_prefetch: don't yet do vprefetch1\n");
          exit(-1);
          break;
       default:
          fprintf(stderr,"libxsmm_instruction_prefetch: Strange prefetch instruction: %u\n",i_prefetch_instr);
          exit(-1);
          break;
    }
    if ( (i_gp_reg_base < LIBXSMM_X86_GP_REG_R8) &&
         (i_gp_reg_idx==LIBXSMM_X86_GP_REG_UNDEF) )
    {
       /* prefetcht1 (%rax) */
       buf[i++] = 0x0f;
       buf[i++] = 0x18;
       buf[i++] = (unsigned char)(0x10 + l_instype + i_gp_reg_base);
       l_place = i-1;
       if ( i_gp_reg_base == LIBXSMM_X86_GP_REG_RSP )
       {
          buf[i++] = 0x24;
       }
    } else if (i_gp_reg_idx==LIBXSMM_X86_GP_REG_UNDEF) {
       /* prefetcht1 (%r8) */
       buf[i++] = 0x41;
       buf[i++] = 0x0f;
       buf[i++] = 0x18;
       buf[i++] = (unsigned char)(0x10 + l_instype + i_gp_reg_base - 8);
       l_place = i-1;
       if ( i_gp_reg_base == LIBXSMM_X86_GP_REG_R12 )
       {
          buf[i++] = 0x24;
       }
    } else { /* two GP regs are being used */
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
          buf[i++] = (unsigned char)(0x41 + l_first);
       }
       buf[i++] = 0x0f;
       buf[i++] = 0x18;
       buf[i++] = (unsigned char)(0x14 + l_instype);
       l_place = i - 1;
       buf[i++] = (unsigned char)(0x00 + l_last);
    }
    l_sizereg = 1;
    if ( ( (i_gp_reg_base%8) == 5) && (i_displacement==0) )
    {
       /* Registers like rbp/r13 when you have a displacement of 0, we need
          force the single byte of zero to appear. */
       l_forced_offset = 1;
    }
    i += add_offset ( l_place, i, i_displacement, l_forced_offset, l_sizereg, buf );

    io_generated_code->code_size = i;
    /* *loc = i; */
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_base_name[4];
    char l_instr_name[16];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base_name, 3 );
    libxsmm_get_x86_instr_name( i_prefetch_instr, l_instr_name, 15 );

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s)\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s)\n", l_instr_name, i_displacement, l_gp_reg_base_name );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_alu_imm( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_alu_instr,
                                      const unsigned int      i_gp_reg_number,
                                      const int               i_immediate ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    /* unsigned int l_maxsize = io_generated_code->buffer_size; */
    /* unsigned int l_maxsize = 1024; */
    int l_first = 0;
    int l_second = 0;
    int l_third = 0;
    int l_reg0 = 0;
    int l_extra = 0;

    switch ( i_alu_instr ) {
       case LIBXSMM_X86_INSTR_ADDQ:
          break;
       case LIBXSMM_X86_INSTR_SUBQ:
          l_second += 0x28;
          l_third += 0x28;
          break;
       case LIBXSMM_X86_INSTR_MOVQ:
          l_second += 0x46;
          l_extra += 0x46;
          break;
       case LIBXSMM_X86_INSTR_CMPQ:
          l_second += 0x38;
          l_third += 0x38;
          break;
       default:
          fprintf(stderr,"libxsmm_instruction_alu_imm: Unknown instruction type: %u\n",i_alu_instr);
          exit(-1);
    }
    if ( (i_gp_reg_number > 7) && (i_gp_reg_number <= 15) )
    {
       l_first += 1;
       l_reg0 = i_gp_reg_number - 8;
    } else {
       l_reg0 = i_gp_reg_number;
    }
    if ( (i_immediate <= 127) && (i_immediate >= -128) &&
         (i_alu_instr!=LIBXSMM_X86_INSTR_MOVQ) )
    {
       /* one byte (even for 0!) - but never for MOVQ */
       buf[i++] = (unsigned char)(0x48 + l_first);
       buf[i++] = 0x83;
       buf[i++] = (unsigned char)(0xc0 + l_third + l_reg0);
       buf[i++] = (unsigned char)(i_immediate);
    } else {
       /* four bytes */
       unsigned char *l_cptr = (unsigned char *) &i_immediate;
       buf[i++] = (unsigned char)(0x48 + l_first);
       if ( i_gp_reg_number==0 && (i_alu_instr!=LIBXSMM_X86_INSTR_MOVQ) )
       {
          /* special case for %rax! */
          buf[i++] = (unsigned char)(0x05 + l_second);
       } else {
          buf[i++] = (unsigned char)(0x81 + l_extra);
          buf[i++] = (unsigned char)(0xc0 + l_third + l_reg0);
       }
       buf[i++] = l_cptr[0];
       buf[i++] = l_cptr[1];
       buf[i++] = l_cptr[2];
       buf[i++] = l_cptr[3];
    }

    io_generated_code->code_size = i;
    /* *loc = i; */
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];
    char l_instr_name[16];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name, 3 );
    libxsmm_get_x86_instr_name( i_alu_instr, l_instr_name, 15 );

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s $%i, %%%%%s\\n\\t\"\n", l_instr_name, i_immediate, l_gp_reg_name );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%i, %%%s\n", l_instr_name, i_immediate, l_gp_reg_name );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_alu_reg( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_alu_instr,
                                      const unsigned int      i_gp_reg_number_src,
                                      const unsigned int      i_gp_reg_number_dest) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    /* unsigned int l_maxsize = io_generated_code->buffer_size; */
    /* unsigned int l_maxsize = 1024; */
    int l_first = 0;
    int l_second = 0;
    int l_reg0 = 0;
    int l_reg1 = 0;

    switch ( i_alu_instr ) {
       case LIBXSMM_X86_INSTR_ADDQ:
          break;
       case LIBXSMM_X86_INSTR_SUBQ:
          l_second += 0x28;
          break;
       case LIBXSMM_X86_INSTR_MOVQ:
          l_second += 0x88;
          break;
       case LIBXSMM_X86_INSTR_CMPQ:
          l_second += 0x38;
          break;
       default:
          fprintf(stderr,"libxsmm_instruction_alu_reg: Not sure what instruction you have in mind: %u\n",i_alu_instr);
          exit(-1);
    }
    if ( (i_gp_reg_number_src > 7) && (i_gp_reg_number_src <=15) )
    {
       l_first += 4;
       l_reg0 = i_gp_reg_number_src - 8;
    } else {
       l_reg0 = i_gp_reg_number_src;
    }
    if ( (i_gp_reg_number_dest > 7) && (i_gp_reg_number_dest <=15) )
    {
       l_first += 1;
       l_reg1 = i_gp_reg_number_dest - 8;
    } else {
       l_reg1 = i_gp_reg_number_dest;
    }

    buf[i++] = (unsigned char)(0x48 + l_first);
    buf[i++] = (unsigned char)(0x01 + l_second);
    buf[i++] = (unsigned char)(0xc0 + 8*l_reg0 + l_reg1);

    io_generated_code->code_size = i;
    /* *loc = i; */
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name_src[4];
    char l_gp_reg_name_dest[4];
    char l_instr_name[16];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_number_src, l_gp_reg_name_src, 3 );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_number_dest, l_gp_reg_name_dest, 3 );
    libxsmm_get_x86_instr_name( i_alu_instr, l_instr_name, 15 );

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%s, %%%%%s\\n\\t\"\n", l_instr_name, l_gp_reg_name_src, l_gp_reg_name_dest );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%s, %%%s\n", l_instr_name, l_gp_reg_name_src, l_gp_reg_name_dest );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_push_reg( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_gp_reg_number ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int l_reg0 = 0;

    if ( l_maxsize - i < 2 )
    {
      fprintf(stderr,"libxsmm_instruction_push_reg: push instructions need up to 2 bytes\n");
      exit(-1);
    }
    if ( /*i_gp_reg_number < 0 ||*/ i_gp_reg_number > 15 ) {
      fprintf(stderr,"libxsmm_instruction_push_reg: invalid register\n");
      exit(-1);
    }

    /* determine register encoding */
    if ( (i_gp_reg_number > 7) && (i_gp_reg_number <=15) )
    {
       l_reg0 = i_gp_reg_number - 8;
       buf[i++] = (unsigned char)(0x41);
    } else {
       l_reg0 = i_gp_reg_number;
    }
    buf[i++] = (unsigned char)(0x50 + l_reg0);

    io_generated_code->code_size = i;
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name, 3 );

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"pushq %%%%%s\\n\\t\"\n", l_gp_reg_name );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       pushq %%%s\n", l_gp_reg_name );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_pop_reg( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_gp_reg_number ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int l_reg0 = 0;

    if ( l_maxsize - i < 2 )
    {
      fprintf(stderr,"libxsmm_instruction_pop_reg: pop instructions need up to 2 bytes\n");
      exit(-1);
    }
    if ( /*i_gp_reg_number < 0 ||*/ i_gp_reg_number > 15 ) {
      fprintf(stderr,"libxsmm_instruction_pop_reg: invalid register\n");
      exit(-1);
    }

    /* determine register encoding */
    if ( (i_gp_reg_number > 7) && (i_gp_reg_number <=15) )
    {
       l_reg0 = i_gp_reg_number - 8;
       buf[i++] = (unsigned char)(0x41);
    } else {
       l_reg0 = i_gp_reg_number;
    }
    buf[i++] = (unsigned char)(0x50 + l_reg0 + 8);

    io_generated_code->code_size = i;
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name, 3 );

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"popq %%%%%s\\n\\t\"\n", l_gp_reg_name );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       popq %%%s\n", l_gp_reg_name );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_mask_move( libxsmm_generated_code* io_generated_code,
                                        const unsigned int      i_mask_instr,
                                        const unsigned int      i_gp_reg_number,
                                        const unsigned int      i_mask_reg_number ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /* unsigned int l_maxsize = 1024; */

    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"libxsmm_instruction_mask_move: Most instructions need at most 20 bytes\n");
       exit(-1);
    }
    switch ( i_mask_instr ) {
       case LIBXSMM_X86_INSTR_KMOVW:
          break;
       default:
          fprintf(stderr,"libxsmm_instruction_mask_move: Strange kmov instruction");
          exit(-1);
          break;
    }
    if ( i_mask_reg_number > 7 )
    {
       fprintf(stderr,"libxsmm_instruction_mask_move: Strange mask number=%u\n",i_mask_reg_number);
       exit(-1);
    }
    if ( (i_gp_reg_number >=8) && (i_gp_reg_number <=15) )
    {
       buf[i++] = 0xc4;
       buf[i++] = 0xc1;
       buf[i++] = 0x78;
       buf[i++] = 0x92;
       buf[i++] = (unsigned char)(0xb8 + i_gp_reg_number + 8*i_mask_reg_number);
    } else {
       buf[i++] = 0xc5;
       buf[i++] = 0xf8;
       buf[i++] = 0x92;
       buf[i++] = (unsigned char)(0xc0 + i_gp_reg_number + 8*i_mask_reg_number);
    }

    io_generated_code->code_size = i;
    /* *loc = i; */
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];
    char l_instr_name[16];
    char l_prefix = '\0';

    libxsmm_get_x86_gp_reg_name( i_gp_reg_number, l_gp_reg_name, 3 );
    libxsmm_get_x86_instr_name( i_mask_instr, l_instr_name, 15 );

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
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%s%c, %%%%k%u\\n\\t\"\n", l_instr_name, l_gp_reg_name, l_prefix, i_mask_reg_number );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%s%c, %%k%u\n", l_instr_name, l_gp_reg_name, l_prefix, i_mask_reg_number );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_mask_compute_reg( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_mask_instr,
                                               const unsigned int      i_mask_reg_number_src_0,
                                               const unsigned int      i_mask_reg_number_src_1,
                                               const unsigned int      i_mask_reg_number_dest ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO-GREG call encoding here */
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /* unsigned int l_maxsize = 1024; */

    if ( l_maxsize - i < 20 )
    {
       fprintf(stderr,"libxsmm_x86_instruction_mask_compute_reg: Most instructions need at most 20 bytes\n");
       exit(-1);
    }
    switch ( i_mask_instr ) {
       case LIBXSMM_X86_INSTR_KXNORW:
          break;
       default:
          fprintf(stderr,"libxsmm_x86_instruction_mask_compute_reg: Strange kmov instruction");
          exit(-1);
          break;
    }
    buf[i++] = 0xc5;
    buf[i++] = (unsigned char)(0xfc - i_mask_reg_number_src_1 * 8);
    buf[i++] = 0x46;
    buf[i++] = (unsigned char)(0xc0 + i_mask_reg_number_src_0 + i_mask_reg_number_dest * 8);

    io_generated_code->code_size = i;
    /* *loc = i; */
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[16];

    libxsmm_get_x86_instr_name( i_mask_instr, l_instr_name, 15 );

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%k%u, %%%%k%u, %%%%k%u\\n\\t\"\n", l_instr_name, i_mask_reg_number_src_0, i_mask_reg_number_src_1, i_mask_reg_number_dest );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%k%u, %%k%u, %%k%u\n", l_instr_name, i_mask_reg_number_src_0, i_mask_reg_number_src_1, i_mask_reg_number_dest );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_register_jump_label( libxsmm_generated_code*     io_generated_code,
                                                  libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  /* check if we still have lable we can jump to */
  if ( io_loop_label_tracker->label_count == 32 ) {
    libxsmm_handle_error( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    int l_lab = io_loop_label_tracker->label_count;
    io_loop_label_tracker->label_count++;
    io_loop_label_tracker->label_address[l_lab] = io_generated_code->code_size;
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] = io_loop_label_tracker->label_count;

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%u:\\n\\t\"\n", io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %u:\n", io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    io_loop_label_tracker->label_count++;
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_jump_back_to_label( libxsmm_generated_code*     io_generated_code,
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
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int l_lab = --io_loop_label_tracker->label_count;
    int l_val = io_loop_label_tracker->label_address[l_lab];
    int l_dist;

    if ( l_maxsize - i < 6 )
    {
       fprintf(stderr,"libxsmm_instruction_jump_back_to_label: Our jump instructions need at most 6 bytes\n");
       exit(-1);
    }
    if ( l_val < i + 2 )
    {
       l_dist = -1*(i+2-l_val); /* assume 1-byte jump initially */
       if ( l_dist >= -128 )    /* can it be done in a single byte? */
       {
          /* Single byte back jump */
          buf[i++] = 0x7c;
          buf[i++] = (unsigned char) l_dist;
       } else {
          unsigned char *l_cptr = (unsigned char *) &l_dist;
          /* 4-byte back jump */
          l_dist = -1*(i+6-l_val);  /* recalc the distance assuming 4-bytes */
          buf[i++] = 0x0f;
          buf[i++] = 0x8c;
          buf[i++] = l_cptr[0];
          buf[i++] = l_cptr[1];
          buf[i++] = l_cptr[2];
          buf[i++] = l_cptr[3];
       }
    } else {
       fprintf(stderr,"libxsmm_instruction_jump_back_to_label: Looks like we're attempting a forward jump\n");
       exit(-1);
    }
    io_generated_code->code_size = i;
    /* *loc = i; */
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_jmp_instr, l_instr_name, 15 );

    io_loop_label_tracker->label_count--;

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %ub\\n\\t\"\n", l_instr_name, io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %ub\n", l_instr_name, io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] = 0;
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_open_stream( libxsmm_generated_code*       io_generated_code,
                                          const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                          const char*                   i_arch,
                                          unsigned int                  i_prefetch) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    unsigned char* l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (l_max_size < (l_code_size + 9)) {
      fprintf(stderr, "libxsmm_generator_gemm_x86_open_instruction_stream: Jit buffer too small\n!");
      exit(-1);
    }

    /* check for a valid register allocation for input pointers */
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

    /* push callee save registers */
    if ( (strcmp(i_arch, "wsm") == 0) ||
         (strcmp(i_arch, "snb") == 0) ||
         (strcmp(i_arch, "hsw") == 0)    ) {
      /* handle m-loop */
      if ( i_gp_reg_mapping->gp_reg_mloop == LIBXSMM_X86_GP_REG_RBX ) {
        l_code_buffer[l_code_size++] = 0x53;
      } else if ( i_gp_reg_mapping->gp_reg_mloop == LIBXSMM_X86_GP_REG_R12 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x54;
      } else if ( i_gp_reg_mapping->gp_reg_mloop == LIBXSMM_X86_GP_REG_R13 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x55;
      } else if ( i_gp_reg_mapping->gp_reg_mloop == LIBXSMM_X86_GP_REG_R14 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x56;
      } else if ( i_gp_reg_mapping->gp_reg_mloop == LIBXSMM_X86_GP_REG_R15 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x57;
      } else {}

      /* handle n-loop */
      if ( i_gp_reg_mapping->gp_reg_nloop == LIBXSMM_X86_GP_REG_RBX ) {
        l_code_buffer[l_code_size++] = 0x53;
      } else if ( i_gp_reg_mapping->gp_reg_nloop == LIBXSMM_X86_GP_REG_R12 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x54;
      } else if ( i_gp_reg_mapping->gp_reg_nloop == LIBXSMM_X86_GP_REG_R13 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x55;
      } else if ( i_gp_reg_mapping->gp_reg_nloop == LIBXSMM_X86_GP_REG_R14 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x56;
      } else if ( i_gp_reg_mapping->gp_reg_nloop == LIBXSMM_X86_GP_REG_R15 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x57;
      } else {}

      /* handle k-loop */
      if ( i_gp_reg_mapping->gp_reg_kloop == LIBXSMM_X86_GP_REG_RBX ) {
        l_code_buffer[l_code_size++] = 0x53;
      } else if ( i_gp_reg_mapping->gp_reg_kloop == LIBXSMM_X86_GP_REG_R12 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x54;
      } else if ( i_gp_reg_mapping->gp_reg_kloop == LIBXSMM_X86_GP_REG_R13 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x55;
      } else if ( i_gp_reg_mapping->gp_reg_kloop == LIBXSMM_X86_GP_REG_R14 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x56;
      } else if ( i_gp_reg_mapping->gp_reg_kloop == LIBXSMM_X86_GP_REG_R15 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x57;
      } else {}

    } else {
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
    }

    /* update code length */
    io_generated_code->code_size = l_code_size;
  } else if ( io_generated_code->code_type == 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
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
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_mloop, l_gp_reg_name, 3 );
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      }
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_nloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_nloop, l_gp_reg_name, 3 );
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      }
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_kloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_name, 3 );
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      }
    } else {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%rbx\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r12\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r13\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r14\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r15\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    }
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];

    /* loading b pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "  __asm__ __volatile__(\"movq %%0, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading a pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%1, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading c pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_c, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%2, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading b prefetch pointer in assembly */
    if ( i_prefetch == LIBXSMM_PREFETCH_BL2_VIA_C ||
         i_prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_name, 3 );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    /* loading a prefetch pointer in assembly */
    } else if ( i_prefetch == LIBXSMM_PREFETCH_AL2 ||
                i_prefetch == LIBXSMM_PREFETCH_AL2_JPST) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_name, 3 );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    /* loading a and b prefetch pointer in assembly */
    } else if ( i_prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C ||
                i_prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_name, 3 );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_name, 3 );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%4, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else {}
  }

  /* reset loop counters */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_mloop, 0 );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_nloop, 0 );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_mapping->gp_reg_kloop, 0 );
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_close_stream( libxsmm_generated_code*       io_generated_code,
                                           const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                           const char*                   i_arch,
                                           unsigned int                  i_prefetch) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is a very simple System V ABI 64 interfacce */
    unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (l_max_size < (l_code_size + 10)) {
      fprintf(stderr, "libxsmm_generator_gemm_x86_close_instruction_stream: Jit buffer too small!\n" );
      exit(-1);
    }

    /* check for a valid register allocation for input pointers */
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

    /* pop callee save registers */
    if ( (strcmp(i_arch, "wsm") == 0) ||
         (strcmp(i_arch, "snb") == 0) ||
         (strcmp(i_arch, "hsw") == 0)    ) {
      /* handle k-loop */
      if ( i_gp_reg_mapping->gp_reg_kloop == LIBXSMM_X86_GP_REG_RBX ) {
        l_code_buffer[l_code_size++] = 0x5b;
      } else if ( i_gp_reg_mapping->gp_reg_kloop == LIBXSMM_X86_GP_REG_R12 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5c;
      } else if ( i_gp_reg_mapping->gp_reg_kloop == LIBXSMM_X86_GP_REG_R13 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5d;
      } else if ( i_gp_reg_mapping->gp_reg_kloop == LIBXSMM_X86_GP_REG_R14 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5e;
      } else if ( i_gp_reg_mapping->gp_reg_kloop == LIBXSMM_X86_GP_REG_R15 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5f;
      } else {}

      /* handle n-loop */
      if ( i_gp_reg_mapping->gp_reg_nloop == LIBXSMM_X86_GP_REG_RBX ) {
        l_code_buffer[l_code_size++] = 0x5b;
      } else if ( i_gp_reg_mapping->gp_reg_nloop == LIBXSMM_X86_GP_REG_R12 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5c;
      } else if ( i_gp_reg_mapping->gp_reg_nloop == LIBXSMM_X86_GP_REG_R13 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5d;
      } else if ( i_gp_reg_mapping->gp_reg_nloop == LIBXSMM_X86_GP_REG_R14 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5e;
      } else if ( i_gp_reg_mapping->gp_reg_nloop == LIBXSMM_X86_GP_REG_R15 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5f;
      } else {}

      /* handle m-loop */
      if ( i_gp_reg_mapping->gp_reg_mloop == LIBXSMM_X86_GP_REG_RBX ) {
        l_code_buffer[l_code_size++] = 0x5b;
      } else if ( i_gp_reg_mapping->gp_reg_mloop == LIBXSMM_X86_GP_REG_R12 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5c;
      } else if ( i_gp_reg_mapping->gp_reg_mloop == LIBXSMM_X86_GP_REG_R13 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5d;
      } else if ( i_gp_reg_mapping->gp_reg_mloop == LIBXSMM_X86_GP_REG_R14 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5e;
      } else if ( i_gp_reg_mapping->gp_reg_mloop == LIBXSMM_X86_GP_REG_R15 ) {
        l_code_buffer[l_code_size++] = 0x41;
        l_code_buffer[l_code_size++] = 0x5f;
      } else {}

    } else {
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
    }

    /* @TODO: I don't know if this is the correct placement in the generation process */
    l_code_buffer[l_code_size++] = 0xc3;

    /* update code length */
    io_generated_code->code_size = l_code_size;
  } else if ( io_generated_code->code_type == 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];

    if ( (strcmp(i_arch, "wsm") == 0) ||
         (strcmp(i_arch, "snb") == 0) ||
         (strcmp(i_arch, "hsw") == 0)    ) {
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_kloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_name, 3 );
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      }
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_nloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_nloop, l_gp_reg_name, 3 );
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      }
      if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_mloop ) ) {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_mloop, l_gp_reg_name, 3 );
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%%s\n", l_gp_reg_name );
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      }
    } else {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r15\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r14\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r13\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r12\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%rbx\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
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
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       retq\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    char l_new_code[1024];
    int l_max_code_length = 1023;
    int l_code_length = 0;
    char l_gp_reg_a[4];
    char l_gp_reg_b[4];
    char l_gp_reg_c[4];
    char l_gp_reg_pre_a[4];
    char l_gp_reg_pre_b[4];
    char l_gp_reg_mloop[4];
    char l_gp_reg_nloop[4];
    char l_gp_reg_kloop[4];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a, l_gp_reg_a, 3 );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b, l_gp_reg_b, 3 );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_c, l_gp_reg_c, 3 );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_pre_a, 3 );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_pre_b, 3 );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_mloop, l_gp_reg_mloop, 3 );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_nloop, l_gp_reg_nloop, 3 );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_kloop, 3 );

    if ( i_prefetch == LIBXSMM_PREFETCH_BL2_VIA_C ||
         i_prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
      if ( (strcmp(i_arch, "wsm") == 0) ||
           (strcmp(i_arch, "snb") == 0) ||
           (strcmp(i_arch, "hsw") == 0)    ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(B_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_b, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(B_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else if ( i_prefetch == LIBXSMM_PREFETCH_AL2 ||
                i_prefetch == LIBXSMM_PREFETCH_AL2_JPST) {
      if ( (strcmp(i_arch, "wsm") == 0) ||
           (strcmp(i_arch, "snb") == 0) ||
           (strcmp(i_arch, "hsw") == 0)    ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else if ( i_prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C ||
                i_prefetch == LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST) {
      if ( (strcmp(i_arch, "wsm") == 0) ||
           (strcmp(i_arch, "snb") == 0) ||
           (strcmp(i_arch, "hsw") == 0)    ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch), \"m\"(B_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_pre_b, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch), \"m\"(B_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else {
      if ( (strcmp(i_arch, "wsm") == 0) ||
           (strcmp(i_arch, "snb") == 0) ||
           (strcmp(i_arch, "hsw") == 0)    ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(B), \"m\"(A), \"m\"(C) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(B), \"m\"(A), \"m\"(C) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_open_stream_convolution( libxsmm_generated_code*                   io_generated_code,
                                                      const unsigned int                        i_gp_reg_input,
                                                      const unsigned int                        i_gp_reg_weight,
                                                      const unsigned int                        i_gp_reg_output,
                                                      const unsigned int                        i_gp_reg_input_pf,
                                                      const unsigned int                        i_gp_reg_weight_pf,
                                                      const unsigned int                        i_gp_reg_output_pf,
                                                      const char*                               i_arch ) {
  LIBXSMM_UNUSED(i_arch);
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    unsigned char* l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (l_max_size < (l_code_size + 9)) {
      fprintf(stderr, "libxsmm_x86_instruction_open_stream_conv: Jit buffer too small\n!");
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
    int l_max_code_length = 511;
    int l_code_length = 0;

    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%rbx\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r12\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r13\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r14\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r15\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];

    /* loading input pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_input, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "  __asm__ __volatile__(\"movq %%0, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading weight pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_weight, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%1, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading output pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_output, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%2, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading input pf pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_input_pf, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading weight pf pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_weight_pf, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%4, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading output pf pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_output_pf, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%5, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_INTERNAL_API_DEFINITION
void libxsmm_x86_instruction_close_stream_convolution( libxsmm_generated_code*       io_generated_code,
                                                       const char*                   i_arch) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is a very simple System V ABI 64 interfacce */
    unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (l_max_size < (l_code_size + 10)) {
      fprintf(stderr, "libxsmm_generator_dense_x86_close_instruction_stream: Jit buffer too small!\n" );
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
    /* @TODO: I don't know if this is the correct placement in the generation process */
    l_code_buffer[l_code_size++] = 0xc3;

    /* update code length */
    io_generated_code->code_size = l_code_size;
  } else if ( io_generated_code->code_type == 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r15\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r14\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r13\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%r12\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%rbx\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* @TODO: I don't know if this is the correct placement in the generation process */
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       retq\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    char l_new_code[1024];
    int l_max_code_length = 1023;
    int l_code_length = 0;

    if ( (strcmp(i_arch, "wsm") == 0) ||
         (strcmp(i_arch, "snb") == 0) ||
         (strcmp(i_arch, "hsw") == 0)    ) {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(inputptr), \"m\"(weightptr), \"m\"(outputptr), \"m\"(inputpfptr), \"m\"(weightpfptr), \"m\"(outputpfptr) : \"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n");
    } else {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(inputptr), \"m\"(weightptr), \"m\"(outputptr), \"m\"(inputpfptr), \"m\"(weightpfptr), \"m\"(outputpfptr) : \"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


