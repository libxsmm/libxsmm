/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Greg Henry (Intel Corp.)
******************************************************************************/
#include "generator_x86_instructions.h"
#include "generator_common.h"


/**
 * This routine is for the jit code. All offsets/displacements have similar
 * byte patterns, so this is used for all of them.
 */
LIBXSMM_API_INLINE
int internal_x86_instructions_add_offset(const unsigned int i_place1,
  const unsigned int i_place2,
  const int i_offset,
  const unsigned int i_forced,
  const int i_sizereg,
  unsigned char *buf)
{
  if ((i_offset == 0) && (i_forced == 0)) return (0);
  else if (((i_offset%i_sizereg) == 0) &&
    (i_offset / i_sizereg <= 127) &&
    (i_offset / i_sizereg >= -128))
  {
    buf[i_place1] += 0x40;
    buf[i_place2] = (unsigned char)(i_offset / i_sizereg);
    return (1);
  }
  else {
    unsigned char *l_cptr = (unsigned char *)&i_offset;
    buf[i_place1] += 0x80;
    buf[i_place2] = l_cptr[0];
    buf[i_place2 + 1] = l_cptr[1];
    buf[i_place2 + 2] = l_cptr[2];
    buf[i_place2 + 3] = l_cptr[3];
    return (4);
  }
}

/**
 * This routine is for the jump jit code. All jumps have similar patterns.
 * Back jumps can be computed immediately because the source and dest is known
 * Forward jumps can be estimated as 4-byte jumps taking 5 or 6 bytes in total
 * i_src_location: location of the start of the jump instruction. It's passed
 *     in as it may have nothing to do with the last location coded in our jit
 *     stream. For backward jumps, it's probably io_generated_code->code_size
 * i_dest_location: location of the start of the target destination we are
 *     jumping to, or -1 if it's a forward jump and currently unknown
 * i_jmp_instr is one of the jump instructions we support
 * This function returns the number of bytes it uses, or 0 if it fails
 */
LIBXSMM_API_INLINE
int internal_x86_jumping( libxsmm_generated_code* io_generated_code,
                          int i_src_location,
                          int i_dest_location,
                          const unsigned int i_jmp_instr )
{
  unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
  int l_jmptype;
  int l_dist;
  unsigned char *l_cptr = (unsigned char *) &l_dist;

  /* check that we just handle a valid jump */
  switch ( i_jmp_instr ) {
     case LIBXSMM_X86_INSTR_JL:
        l_jmptype = 0x7c;
        break;
     case LIBXSMM_X86_INSTR_JE:
     case LIBXSMM_X86_INSTR_JZ:
        l_jmptype = 0x74;
        break;
     case LIBXSMM_X86_INSTR_JG:
        l_jmptype = 0x7F;
        break;
     case LIBXSMM_X86_INSTR_JNE:
     case LIBXSMM_X86_INSTR_JNZ:
        l_jmptype = 0x75;
        break;
     case LIBXSMM_X86_INSTR_JGE:
        l_jmptype = 0x7D;
        break;
     case LIBXSMM_X86_INSTR_JLE:
        l_jmptype = 0x7E;
        break;
     case LIBXSMM_X86_INSTR_JMP:
        l_jmptype = 0xEB;
        break;
     default:
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUPPORTED_JUMP );
        return 0;
  }
  /* The jmp instruction better be somewhere valid in the code */
  if ( i_src_location < 0 )
  {
     fprintf(stderr,"Bogus source location for internal jumping routine: %d\n",i_src_location );
     exit(-1);
  }
  /* Make sure i_src_location is no bigger than the end of the code */
  if ( (unsigned int)i_src_location > io_generated_code->code_size )
  {
     fprintf(stderr,"How can the source of the jump itself be an instruction far beyond where we've jitted? Something is really strange here src=%i loc=%u\n",i_src_location,io_generated_code->code_size);
     exit(-1);
  }

  if ( i_dest_location < 0 )
  {
     /* Must be a forward jump and we don't yet know it's dest location */
     if ( i_jmp_instr == LIBXSMM_X86_INSTR_JMP ) {
        buf[i_src_location] = 0xe9;
        /* FIll-in zeros for now, this routine has to be called again: */
        buf[i_src_location+1] = 0x00;
        buf[i_src_location+2] = 0x00;
        buf[i_src_location+3] = 0x00;
        buf[i_src_location+4] = 0x00;
        return 5;
     } else {
        buf[i_src_location] = 0x0f;
        buf[i_src_location+1] = (unsigned char)(l_jmptype + 0x10);
        /* FIll-in zeros for now, this routine has to be called again: */
        buf[i_src_location+2] = 0x00;
        buf[i_src_location+3] = 0x00;
        buf[i_src_location+4] = 0x00;
        buf[i_src_location+5] = 0x00;
        return 6;
     }
  }

  /* Make sure we aren't trying to jump to the same location as the original jump instruction */
  if ( i_src_location==i_dest_location || (i_src_location==i_dest_location+1) )
  {
     fprintf(stderr,"i_src_location=%d is physically too close to i_dest_location=%d\n",i_src_location,i_dest_location);
     exit(-1);
  }

  if ( i_src_location > i_dest_location )
  {
     /* Must be a backward jump */
     l_dist = -1*(i_src_location+2-i_dest_location); /* assume 1-byte */
     if ( l_dist >= -128 ) /* can it be done in 1-byte? */
     {
        /* Single byte back jump */
        buf[i_src_location]   = (unsigned char)l_jmptype;
        buf[i_src_location+1] = (unsigned char)l_dist;
        return 2;
     } else {
        /* 4-byte back jump */
        if ( i_jmp_instr != LIBXSMM_X86_INSTR_JMP ) {
           /* l_cptr better point to l_dist and l_dist needs to be recalculated */
           l_dist = -1*(i_src_location+6-i_dest_location);
           buf[i_src_location]   = 0x0f;
           buf[i_src_location+1] = (unsigned char)(l_jmptype + 0x10);
           buf[i_src_location+2] = l_cptr[0];
           buf[i_src_location+3] = l_cptr[1];
           buf[i_src_location+4] = l_cptr[2];
           buf[i_src_location+5] = l_cptr[3];
           return 6;
        } else {
           /* l_cptr better point to l_dist and l_dist needs to be recalculated */
           l_dist = -1*(i_src_location+5-i_dest_location);
           buf[i_src_location]   = 0xE9;
           buf[i_src_location+1] = l_cptr[0];
           buf[i_src_location+2] = l_cptr[1];
           buf[i_src_location+3] = l_cptr[2];
           buf[i_src_location+4] = l_cptr[3];
           return 5;
        }
     }
  } else {
     /* Must be a 4 or 5 byte forward jump with all locations known */
     if ( i_jmp_instr == LIBXSMM_X86_INSTR_JMP ) {
        /* l_cptr better point to l_dist and l_dist needs to be recalculated */
        l_dist = (i_dest_location-i_src_location-5);
        buf[i_src_location] = 0xe9;
        buf[i_src_location+1] = l_cptr[0];
        buf[i_src_location+2] = l_cptr[1];
        buf[i_src_location+3] = l_cptr[2];
        buf[i_src_location+4] = l_cptr[3];
        return 5;
     } else {
        /* l_cptr better point to l_dist and l_dist needs to be recalculated */
        l_dist = (i_dest_location-i_src_location-6);
        buf[i_src_location] = 0x0f;
        buf[i_src_location+1] = (unsigned char)(l_jmptype + 0x10);
        buf[i_src_location+2] = l_cptr[0];
        buf[i_src_location+3] = l_cptr[1];
        buf[i_src_location+4] = l_cptr[2];
        buf[i_src_location+5] = l_cptr[3];
        return 6;
     }
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_mask_move( libxsmm_generated_code* io_generated_code,
                                     const unsigned int      i_vmove_instr,
                                     const unsigned int      i_gp_reg_base,
                                     const unsigned int      i_gp_reg_idx,
                                     const unsigned int      i_scale,
                                     const int               i_displacement,
                                     const char              i_vector_name,
                                     const unsigned int      i_vec_reg_number_0,
                                     const unsigned int      i_vec_reg_mask_0,
                                     const unsigned int      i_is_store )
{
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /* unsigned int l_maxsize = 1024; */
    int l_regbas0 = i_gp_reg_base % 8;
    int l_gp8     = ((i_gp_reg_base > 7)&&(i_gp_reg_base<=15)?1:0);
    int l_regidx  = 0;
    int l_ix8     = ((i_gp_reg_idx > 7)&&(i_gp_reg_idx<=15)?1:0);
    int l_vecval0 = i_vec_reg_number_0 % 8;
    int l_vecgrp0 = i_vec_reg_number_0 / 8;
    int l_oddgrp0 = ((l_vecgrp0 % 2)==1);
    int l_vecval1 = i_vec_reg_mask_0 % 8;
    int l_vecgrp1 = i_vec_reg_mask_0 / 8;
    int l_oddgrp1 = ((l_vecgrp1 % 2)==1);
    int l_sca=0;
    int l_inst = 0;
    int l_place1;

    if ( /*(i_gp_reg_idx>=0) &&*/ i_gp_reg_idx<=15 ) l_regidx = i_gp_reg_idx % 8;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }

    if (i_scale==2) l_sca=0x40;
    else if (i_scale==4) l_sca=0x80;
    else if (i_scale==8) l_sca=0xc0;

    if ( (i_vector_name != 'y') && (i_vector_name != 'Y') )
    {
       fprintf(stderr, "libxsmm_instruction_vec_mask_move only works with i_vector_name as y for ymm* registers\n");
       exit(-1);
    }

    switch ( i_vmove_instr ) {
       case LIBXSMM_X86_INSTR_VMASKMOVPD:
          if ( i_is_store == 0 ) l_inst= 0x01; else l_inst= 0x03;
          break;
       case LIBXSMM_X86_INSTR_VMASKMOVPS:
          if ( i_is_store == 0 ) l_inst= 0x00; else l_inst= 0x02;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_mask_move: Exactly what sort of instructions are you using?\n");
          exit(-1);
    }

    buf[i++] = (unsigned char)(0xc4);
    buf[i++] = (unsigned char)(0xe2 - l_gp8 * 0x20 - l_ix8 * 0x40 - l_oddgrp0 * 0x80);
    buf[i++] = (unsigned char)(0x7d - l_oddgrp1 * 0x40 - l_vecval1*8);
    buf[i++] = (unsigned char)(0x2c + l_inst);
    l_place1 = i;
    if ( /*(i_gp_reg_idx>=0) &&*/ i_gp_reg_idx<=15 )
    {
       buf[i++] = (unsigned char)(0x04 + l_vecval0*8);
       buf[i++] = (unsigned char)(l_sca + l_regbas0 + l_regidx*8);
    } else {
       buf[i++] = (unsigned char)(l_regbas0 + l_vecval0*8);
    }

    i += internal_x86_instructions_add_offset( l_place1, i, i_displacement, 0, 1, buf );

    io_generated_code->code_size = i;
    /* *loc = i; */
  } else {
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_move( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_instruction_set,
                                       const unsigned int      i_vmove_instr,
                                       const unsigned int      i_gp_reg_base,
                                       const unsigned int      i_gp_reg_idx,
                                       const unsigned int      i_scale,
                                       const int               i_displacement,
                                       const char              i_vector_name,
                                       const unsigned int      i_vec_reg_number_0,
                                       const unsigned int      i_mask_reg_number,
                                       const unsigned int      i_use_zero_masking,
                                       const unsigned int      i_is_store )
{
  if ( (i_is_store == 0) && ( (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVNTPD) ||
                              (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVNTPS) ||
                              (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVNTDQ)   )) {
    fprintf(stderr, "libxsmm_instruction_vec_move: streaming stores are only available when setting storing option to true!\n");
    exit(-1);
  }

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
    int l_scaleadj = 0;
    int l_bytes = 4; /* base number of bytes */
    int l_sse3 = 0;
    int l_insert_extra_byte = 0;
    int l_fpadj = 0;

    if ( (i_vector_name != 'z') && (i_mask_reg_number != 0) )
    {
       fprintf(stderr, "libxsmm_instruction_vec_move: Masking is only enabled with zmm registers!\n");
       exit(-1);
    }
    if ( (i_use_zero_masking != 0) && (i_mask_reg_number != 0) && (i_is_store != 0) )
    {
      fprintf(stderr, "libxsmm_instruction_vec_move: zero-masked store cannot operate on memory destination!\n");
      exit(-1);
    }
    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
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
             fprintf(stderr, "libxsmm_instruction_vec_move: You want to use vmovss without xmm?\n");
             exit(-1);
          }
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_ivectype += 2;
          break;
       case LIBXSMM_X86_INSTR_VMOVSD:
          if ( i_vector_name!='x' )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: You want to use vmovsd without xmm?\n");
             exit(-1);
          }
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_ivectype += 3;
          break;
       case LIBXSMM_X86_INSTR_VPBROADCASTD:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vpbroadcastd not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vpbroadcastd and store?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_move: vpbroadcastq not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vpbroadcastq and store?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_move: vpbroadcastb not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vpbroadcastb and store?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_move: vpbroadcastw not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vpbroadcastw and store?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_move: vmovdqa32 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x01;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          if ( i_is_store == 1 ) l_aligned += 0xf;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQA64:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vmovdqa64 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x81;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          if ( i_is_store == 1 ) l_aligned += 0xf;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQU8:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vmovdqu8 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x03;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          if ( i_is_store == 1 ) l_aligned += 0xf;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQU16:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vmovdqu16 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x83;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          if ( i_is_store == 1 ) l_aligned += 0xf;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQU32:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vmovdqu32 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x02;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          if ( i_is_store == 1 ) l_aligned += 0xf;
          break;
       case LIBXSMM_X86_INSTR_VMOVDQU64:
          l_bytes = 5;
          if ( i_vector_name=='x' || i_vector_name=='y' )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vmovdqu64 not yet implemented for xmm/ymm\n");
             exit(-1);
          }
          l_ivectype2 += 0x82;
          l_penultimate += 0x5f;
          l_num3 += 0x21;
          l_sizereg = 64;
          if ( i_is_store == 1 ) l_aligned += 0xf;
          break;
       case LIBXSMM_X86_INSTR_VMOVNTPD:
          l_bytes = 4;
          if ( i_vector_name=='x' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vmovntpd not yet implemented for xmm\n");
             exit(-1);
          }
          if ( l_num == 1 ) l_ivectype3 += 0x80;
          l_ivectype2 += 0x81;
          l_penultimate += 0x1A;
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVNTPS:
          l_bytes = 4;
          if ( i_vector_name=='x' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vmovntps not yet implemented for xmm\n");
             exit(-1);
          }
          if ( l_num == 1 ) l_ivectype3 += 0x80;
          l_ivectype -= 0x01;
          l_penultimate += 0x1A;
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVNTDQ:
          l_bytes = 4;
          if ( i_vector_name=='x' )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: vmovntdq not yet implemented for xmm\n");
             exit(-1);
          }
          if ( l_num == 1 ) l_ivectype3 += 0x80;
          l_ivectype2 += 0x01;
          l_penultimate += 0xD6;
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VBROADCASTSD:
          l_bytes = 5;
          if ( i_vector_name=='x' )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vbroadcastsd and xmm?\n");
             exit(-1);
          }
          if ( i_is_store == 1 )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vbroadcastsd and stores?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_move: vbroadcastss and stores?\n");
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
       case LIBXSMM_X86_INSTR_VPMOVDW:
          if ( i_vector_name=='x' ) l_ivectype += 1;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_sizereg = 32;
          l_ivectype2 += 0x02;
          l_num2 += 1;
          l_penultimate += 0x22;
          break;
       case LIBXSMM_X86_INSTR_VPMOVDB:
          if ( i_vector_name=='x' ) l_ivectype += 1;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_sizereg = 16;
          l_ivectype2 += 0x02;
          l_num2 += 1;
          l_penultimate += 0x20;
          break;
       case LIBXSMM_X86_INSTR_VPMOVSDB:
          if ( i_vector_name=='x' ) l_ivectype += 1;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_sizereg = 16;
          l_ivectype2 += 0x02;
          l_num2 += 1;
          l_penultimate += 0x10;
          break;
       case LIBXSMM_X86_INSTR_VPMOVUSDB:
          if ( i_vector_name=='x' ) l_ivectype += 1;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_sizereg = 16;
          l_ivectype2 += 0x02;
          l_num2 += 1;
          /* l_penultimate += 0x00;*/
          break;
       case LIBXSMM_X86_INSTR_VPMOVSXWD:
          if ( i_vector_name=='x' ) l_ivectype += 1;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_sizereg = 64;
          l_ivectype2 += 0x81;
          l_num3 += 1;
          l_penultimate += 0x13;
          l_bytes = 5;
          l_wow += 0x20;
          break;
       case LIBXSMM_X86_INSTR_VPMOVZXWD:
          if ( i_vector_name=='x' ) l_ivectype += 1;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_sizereg = 32;
          l_ivectype2 += 0x01;
          l_num3 += 1;
          l_penultimate += 0x23;
          l_bytes = 5;
          l_wow += 0x20;
          l_wow += 0xE1;
          break;
       case LIBXSMM_X86_INSTR_VPMOVSXBD:
          if ( i_vector_name=='x' ) l_ivectype += 1;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_sizereg = 16;
          l_ivectype2 += 0x01;
          l_num3 += 1;
          l_penultimate += 0x11;
          l_bytes = 5;
          l_wow += 0x20;
          l_wow += 0xE1;
          break;
       case LIBXSMM_X86_INSTR_VPMOVZXBD:
          if ( i_vector_name=='x' ) l_ivectype += 1;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_sizereg = 16;
          l_ivectype2 += 0x01;
          l_num3 += 1;
          l_penultimate += 0x21;
          l_bytes = 5;
          l_wow += 0x20;
          l_wow += 0xE1;
          break;
       case LIBXSMM_X86_INSTR_VMOVUPS:
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          if ( i_vector_name!='x' ) l_ivectype -= 1; /* single */
          l_sizereg = 64;
          break;
       case LIBXSMM_X86_INSTR_VMOVDDUP:
          if ( i_is_store == 1 )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: vmovddup and stores?\n");
             exit(-1);
          }
          l_ivectype += 2;
          l_ivectype2 += 0x83;
          if ( l_num == 1 ) l_ivectype3 -= 0x80;
          l_penultimate += 2;
          l_sizereg = 64;
          if ( i_vector_name=='x' ) l_ivectype += 1;
          break;
       case LIBXSMM_X86_INSTR_MOVAPD:
          l_sse3 = 1;
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x18;
          break;
       case LIBXSMM_X86_INSTR_MOVUPD:
          l_sse3 = 1;
          l_insert_extra_byte = 0x66;
          break;
       case LIBXSMM_X86_INSTR_MOVAPS:
          l_sse3 = 1;
          l_fpadj = 0x18;
          break;
       case LIBXSMM_X86_INSTR_MOVUPS:
          l_sse3 = 1;
          break;
       case LIBXSMM_X86_INSTR_MOVSD:
          l_sse3 = 1;
          l_insert_extra_byte = 0xF2;
          break;
       case LIBXSMM_X86_INSTR_MOVSS:
          l_sse3 = 1;
          l_insert_extra_byte = 0xF3;
          break;
       case LIBXSMM_X86_INSTR_MOVDDUP:
          l_sse3 = 1;
          l_insert_extra_byte = 0xF2;
          l_fpadj = 2;
          if ( i_is_store )
          {
             fprintf(stderr,"libxsmm_instruction_vec_move: don't support a store with movddup\n");
             exit(-1);
          }
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_move: unexpected instruction number: %u\n",i_vmove_instr);
          exit(-1);
    }
    switch ( i_vector_name ) {
       case 'x':
          l_sizereg = 1;
          if ( l_num > 1 )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: Are you sure xmm%u exists?\n",i_vec_reg_number_0);
             exit(-1);
          }
          break;
       case 'y':
          l_ivectype += 5;
          l_sizereg = 1;
          if ( l_num > 2 )
          {
             fprintf(stderr, "libxsmm_instruction_vec_move: Are you sure ymm%u exists?\n",i_vec_reg_number_0);
             exit(-1);
          }
          break;
       case 'z':
          l_bytes = 6;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_move: Exactly what sort of fp regs are you using?\n");
          exit(-1);
    }
    if ( i_is_store == 1 )
    {
       l_aligned += 1;
       /*if ( i_use_masking != 0 ) l_maskingoff = i_mask_reg_number;*/
    } else {
       /*The following addition of 0x80 appears broken...
        if ( i_use_masking != 0 ) l_maskingoff = 0x80 + i_mask_reg_number; */
    }
    if ( !l_sse3 )
    {
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

    if ( (i_mask_reg_number > 0) && (i_mask_reg_number <= 127) ) {
      l_maskingoff = i_mask_reg_number;
      if ( i_use_zero_masking != 0 && i_is_store == 0 ) l_maskingoff += 0x80;
    }
    if ( l_num == 0 ) l_vregoffset = 0x90;
    else if ( l_num == 1 ) { l_vregoffset = 0x10; l_vregoffset2 = -0x80; }
    else if ( l_num == 2 ) l_vregoffset = 0x80;
    else if ( l_num == 3 ) l_vregoffset = 0x00;
    if ( (l_iregnum == 5) && (i_displacement==0) )
    {
       /* Registers like rbp/r13 when you have a displacement of 0, we need
          force the single byte of zero to appear. */
       l_forced_offset = 1;
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
       if ( i_scale == 1 ) l_scaleadj = 0x00;
       else if ( i_scale == 2 ) l_scaleadj = 0x40;
       else if ( i_scale == 4 ) l_scaleadj = 0x80;
       else if ( i_scale == 8 ) l_scaleadj = 0xc0;
       else
       {
          fprintf(stderr, "libxsmm_instruction_vec_move: cannot handle i_scale=%u parameter\n", i_scale);
          exit(-1);
       }
       buf[i++] = (unsigned char)(l_scaleadj + l_iregnum + 8*(i_gp_reg_idx%8));
    } else {
       l_place = i;
       buf[i++] = (unsigned char)(0x00 + l_iregnum + 8*l_vregnum);
       if ( l_iregnum == LIBXSMM_X86_GP_REG_RSP )
       {
          buf[i++] = 0x24;
       }
    }
    i += internal_x86_instructions_add_offset( l_place, i, i_displacement, l_forced_offset, l_sizereg, buf );

    io_generated_code->code_size = i;
    /* *loc = i; */

    } else {
        /* SSE3 code */
        int l_vecgrp0 = 0;
        int l_vecval0 = i_vec_reg_number_0 % 8;
        int l_place1=i+2;
        int l_regbas0 = i_gp_reg_base % 8;
        int l_regidx =  i_gp_reg_idx % 8;
        int l_gp8 = ((i_gp_reg_base > 7)&&(i_gp_reg_base<=15)?1:0);
        if ( (i_vec_reg_number_0>=8) && (i_vec_reg_number_0<=15) ) l_vecgrp0=1;
        if ( i_is_store ) l_fpadj++;
        if ( l_insert_extra_byte != 0 )
        {
            buf[i++]= (unsigned char)(l_insert_extra_byte);
            ++l_place1;
        }
        if (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF )
        {
            int l_sse_preamble2 = 64;
            if ( l_gp8 || (l_vecgrp0>=1) )
            {
               if (l_gp8) l_sse_preamble2 += 1;
               if (l_vecgrp0 >=1) l_sse_preamble2 += 4;
               buf[i++] = (unsigned char)(l_sse_preamble2);
               ++l_place1;
            }
            buf[i++] = (unsigned char)(0x0f);
            buf[i++] = (unsigned char)(0x10 + l_fpadj);
            buf[i++] = (unsigned char)(0x00 + l_regbas0 + l_vecval0*8);
            if ( l_regbas0 == 4 ) buf[i++]=0x24;
        } else {
          int l_ix8 = ((i_gp_reg_idx > 7) && (i_gp_reg_idx <= 15) ? 1 : 0);
          int l_sse_preamble2 = 64;
          if ( i_scale == 1 ) l_scaleadj = 0x00;
            else if ( i_scale == 2 ) l_scaleadj = 0x40;
            else if ( i_scale == 4 ) l_scaleadj = 0x80;
            else if ( i_scale == 8 ) l_scaleadj = 0xc0;
            else
            {
               fprintf(stderr, "libxsmm_instruction_vec_move sse3 section: cannot handle i_scale=%u parameter\n", i_scale);
               exit(-1);
            }
            if ( l_gp8 || l_ix8 || (l_vecgrp0>=1) )
            {
                if (l_gp8) l_sse_preamble2 += 1;
                if (l_ix8) l_sse_preamble2 += 2;
                if (l_vecgrp0 >=1) l_sse_preamble2 += 4;
                buf[i++] = (unsigned char)(l_sse_preamble2);
                ++l_place1;
            }
            buf[i++] = (unsigned char)(0x0f);
            buf[i++] = (unsigned char)(0x10 + l_fpadj);
            buf[i++] = (unsigned char)(0x04 + l_vecval0*8);
            buf[i++] = (unsigned char)(0x00 + l_scaleadj + l_regbas0 + l_regidx*8);
        }
        l_forced_offset = 0;
        if ( (l_regbas0 == 5) && (i_displacement==0) )
        {
            l_forced_offset = 1;
        }
        i += internal_x86_instructions_add_offset( l_place1, i, i_displacement, l_forced_offset, l_sizereg, buf );
        io_generated_code->code_size = i;
        /* *loc = i; */
    }
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_base_name[4];
    char l_instr_name[16];
    char l_masking_type[16];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base_name, 3 );
    libxsmm_get_x86_instr_name( i_vmove_instr, l_instr_name, 15 );

    if ( i_use_zero_masking == 0 || i_mask_reg_number == 0 || i_is_store != 0 ) {
      if ( io_generated_code->code_type == 0 ) {
        l_masking_type[0] = (char)0; /* no zero-masking */
      }
    } else {
      if ( io_generated_code->code_type == 0 ) {
        LIBXSMM_SNPRINTF(l_masking_type, 16, "%%{z%%}" );
      } else {
        LIBXSMM_SNPRINTF(l_masking_type, 16, "{z}" );
      }
    }

    if ( (i_instruction_set >= LIBXSMM_X86_AVX512) &&
         (i_mask_reg_number != 0) ) {
      /* build vmovpd/ps/sd/ss instruction, load use */
      if ( i_is_store == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s), %%%%%cmm%u%%{%%%%k%u%%}%s\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0, i_mask_reg_number, l_masking_type );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%%cmm%u{%%k%u}%s\n", l_instr_name, i_displacement, l_gp_reg_base_name, i_vector_name, i_vec_reg_number_0, i_mask_reg_number, l_masking_type );
        }
      } else { /* store */
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %i(%%%%%s)%%{%%%%k%u%%}%s\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name, i_mask_reg_number, l_masking_type );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %i(%%%s) {%%k%u}%s\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_displacement, l_gp_reg_base_name, i_mask_reg_number, l_masking_type );
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

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_convert ( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_instruction_set,
                                                   const unsigned int      i_vec_instr,
                                                   const char              i_vector_name,
                                                   const unsigned int      i_vec_reg_src_0,
                                                   const unsigned int      i_vec_reg_src_1,
                                                   const unsigned int      i_vec_reg_dst,
                                                   const unsigned int      i_shuffle_operand )
{
  LIBXSMM_UNUSED(i_instruction_set);
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size; /* i = *loc; */
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int l_vec0 = 0, l_vec1 = 0, l_second = 0, l_third = 0, l_fourth = 0, l_fifth = 0;
    int l_vecval0, l_vecgrp0, l_oddgrp0, l_2or3grp0;
    int l_vecval1, l_vecgrp1, l_oddgrp1, l_2or3grp1;
    /* these defines are for LIBXSMM_X86_INSTR_VCVTNE2PS2BF16 only: */
    int l_vecvalsrc1, l_vecgrpsrc1, l_oddgrpsrc1, l_2or3grpsrc1;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }
    switch ( i_vector_name ) {
       case 'x':
       case 'y':
          fprintf(stderr, "libxsmm_instruction_vec_compute_convert: the highest register should be zmm: use that\n");
          break;
       case 'z':
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_convert: Unknown sort of fp registers\n");
          exit(-1);
    }

    if ( (i_vec_instr == LIBXSMM_X86_INSTR_VCVTNE2PS2BF16) && (i_vec_reg_src_1 == LIBXSMM_X86_VEC_REG_UNDEF) ) {
      fprintf(stderr, "libxsmm_instruction_vec_compute_convert: VCVTNE2PS2BF16 needs two inputs\n");
      exit(-1);
    }

    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_VCVTDQ2PS:
          l_fifth = 0x48;
          l_vec0 = i_vec_reg_src_0;
          l_vec1 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VPMOVDB:
          l_second = 0x1;
          l_third += 2;
          l_fifth = 0x1E;
          l_vec0 = i_vec_reg_dst;
          l_vec1 = i_vec_reg_src_0;
          break;
       case LIBXSMM_X86_INSTR_VPMOVSDB:
          l_second = 0x1;
          l_third += 2;
          l_fifth = 0xE;
          l_vec0 = i_vec_reg_dst;
          l_vec1 = i_vec_reg_src_0;
          break;
       case LIBXSMM_X86_INSTR_VPMOVUSDB:
          l_second = 0x1;
          l_third += 2;
          l_fifth = -2;
          l_vec0 = i_vec_reg_dst;
          l_vec1 = i_vec_reg_src_0;
          break;
       case LIBXSMM_X86_INSTR_VCVTPS2DQ:
          l_fifth = 0x48;
          l_third += 1;
          l_vec0 = i_vec_reg_src_0;
          l_vec1 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VCVTPS2UDQ:
          l_fifth = 0x66;
          l_vec0 = i_vec_reg_src_0;
          l_vec1 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VCVTPS2PD:
          l_fifth = 0x47;
          l_vec0 = i_vec_reg_src_0;
          l_vec1 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VCVTPS2PH:
          l_second = 2;
          l_third = 1;
          l_fifth = 0x0a;
          l_vec1 = i_vec_reg_src_0;
          l_vec0 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VCVTPH2PS:
          l_second = 1;
          l_third = 1;
          l_vec0 = i_vec_reg_src_0;
          l_vec1 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VPMOVDW:
          l_second = 1;
          l_third = 2;
          l_fifth = 0x20;
          l_vec1 = i_vec_reg_src_0;
          l_vec0 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VPMOVSXWD:
          l_second = 1;
          l_third = 1;
          l_fifth = 0x10;
          l_vec0 = i_vec_reg_src_0;
          l_vec1 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VPMOVZXWD:
          l_second = 1;
          l_third = 1;
          l_fifth = 0x20;
          l_vec0 = i_vec_reg_src_0;
          l_vec1 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VPMOVSXBD:
          l_second = 1;
          l_third = 1;
          l_fifth = 0xE;
          l_vec0 = i_vec_reg_src_0;
          l_vec1 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VPMOVZXBD:
          l_second = 1;
          l_third = 1;
          l_fifth = 0x1E;
          l_vec0 = i_vec_reg_src_0;
          l_vec1 = i_vec_reg_dst;
          break;
       case LIBXSMM_X86_INSTR_VCVTNEPS2BF16:
          l_second = 1;
          l_third = 2;
          l_fifth = 0x5F;
          l_vec1 = i_vec_reg_dst;
          l_vec0 = i_vec_reg_src_0;
          break;
       case LIBXSMM_X86_INSTR_VCVTNE2PS2BF16:
          l_vecvalsrc1 = i_vec_reg_src_1 % 8;
          l_vecgrpsrc1 = i_vec_reg_src_1 / 8;
          l_oddgrpsrc1 = ((l_vecgrpsrc1 % 2)==1);
          l_2or3grpsrc1 = (l_vecgrpsrc1>=2);
          l_second = 1;
          l_third = 3 - l_oddgrpsrc1*0x40 - l_vecvalsrc1*0x08;
          l_fourth = -l_2or3grpsrc1 * 0x08;
          l_fifth = 0x5F;
          l_vec1 = i_vec_reg_dst;
          l_vec0 = i_vec_reg_src_0;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_convert: Unknown instruction type: %u\n", i_vec_instr);
          break;
    }
    l_vecval0 = l_vec0 % 8;
    l_vecgrp0 = l_vec0 / 8;
    l_oddgrp0 = ((l_vecgrp0 % 2)==1);
    l_2or3grp0 = (l_vecgrp0>=2);
    l_vecval1 = l_vec1 % 8;
    l_vecgrp1 = l_vec1 / 8;
    l_oddgrp1 = ((l_vecgrp1 % 2)==1);
    l_2or3grp1 = (l_vecgrp1>=2);

    buf[i++] = (unsigned char)(0x62);
    buf[i++] = (unsigned char)(0xf1 + l_second - l_oddgrp0 * 0x20 - l_oddgrp1 * 0x80 - l_2or3grp0 * 0x40 - l_2or3grp1 * 0x10);
    buf[i++] = (unsigned char)(0x7c + l_third);
    buf[i++] = (unsigned char)(0x48 + l_fourth);
    buf[i++] = (unsigned char)(0x13 + l_fifth);
    buf[i++] = (unsigned char)(0xc0 + l_vecval0 + l_vecval1*8);

    if ( i_vec_instr == LIBXSMM_X86_INSTR_VCVTPS2PH )
    {
       buf[i++] = (unsigned char)(i_shuffle_operand);
    }

    io_generated_code->code_size = i;
    /* *loc = i;  */

  } else {
  }
}


LIBXSMM_API_INTERN
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
    int l_reg0, l_reg1, l_reg2;
    int l_vreg0   = i_vec_reg_number_0;
    int l_vreg1   = i_vec_reg_number_1;
    int l_vreg2   = i_vec_reg_number_2;
    int l_fpadj=0;
    int l_fpadj2=0;
    int l_bytes=4;
    int l_sse = 0;
    int l_insert_extra_byte = 0;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }

    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_VXORPD:
          l_fpadj = -2;
          break;
       case LIBXSMM_X86_INSTR_VMULPD:
          break;
       case LIBXSMM_X86_INSTR_VPERMW:
          l_second += 0x01;
          l_fpadj += 0x34;
          break;
       case LIBXSMM_X86_INSTR_VPERMD:
          l_second += 0x01;
          l_fpadj = -0x23;
          l_fpadj2 = -0x80;
          break;
       case LIBXSMM_X86_INSTR_VUNPCKLPD:
          l_fpadj = -0x45;
          break;
       case LIBXSMM_X86_INSTR_VUNPCKLPS:
          l_fpadj = -0x45;
          if ( (i_vector_name!='z') && (l_vreg0<=15) && (l_vreg1<=15) &&
               (l_vreg2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          break;
       case LIBXSMM_X86_INSTR_VUNPCKHPD:
          l_fpadj = -0x44;
          break;
       case LIBXSMM_X86_INSTR_VUNPCKHPS:
          l_fpadj = -0x44;
          if ( (i_vector_name!='z') && (l_vreg0<=15) && (l_vreg1<=15) &&
               (l_vreg2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          break;
       case LIBXSMM_X86_INSTR_VADDPD:
          l_fpadj = -1;
          break;
       case LIBXSMM_X86_INSTR_VDIVPD:
          l_fpadj = 5;
          break;
       case LIBXSMM_X86_INSTR_VDPBF16PS:
          if ( i_vector_name == 'x' )
          {
             l_fourth -= 0x40;
             if ( l_vreg0 >= 16 ) l_fourth -= 0xc0;
             if ( l_vreg1 >= 16 ) l_fourth -= 0xc0;
          } else if ( i_vector_name == 'y' )
          {
             l_fourth -= 0x20;
             if ( l_vreg0 >= 16 ) l_fourth += 0x20;
             if ( l_vreg1 >= 16 ) l_fourth += 0x20;
          }
          l_bytes = 6;
          l_second += 1;
          l_fpadj = -7;
          l_fpadj2 = 0x81;
          break;
       case LIBXSMM_X86_INSTR_VDIVPS:
          if ( (i_vector_name!='z') && (l_vreg0 <=15) &&
               (l_vreg1<=15) && (l_vreg2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = 5;
          break;
       case LIBXSMM_X86_INSTR_VPANDD:
          if ( (i_vector_name!='z') && (i_vector_name!='Z') )
          {
             fprintf(stderr,"VPANDD in vec_compute_reg expects zmm registers\n");
             exit(-1);
          }
          l_fpadj2 = -0x80;
          l_fpadj = 0x82;
          break;
       case LIBXSMM_X86_INSTR_VPANDQ:
          if ( (i_vector_name!='z') && (i_vector_name!='Z') )
          {
             fprintf(stderr,"VPANDQ in vec_compute_reg expects zmm registers\n");
             exit(-1);
          }
          l_fpadj2 = 0;
          l_fpadj = 0x82;
          break;
       case LIBXSMM_X86_INSTR_VMAXPD:
          if ( (i_vector_name!='z') && (i_vector_name!='Z') )
          {
             fprintf(stderr,"VMAXPD in vec_compute_reg expects zmm registers\n");
             exit(-1);
          }
          l_fpadj2 = 0;
          l_fpadj = 6;
          break;
       case LIBXSMM_X86_INSTR_VMAXPS:
          if ( (i_vector_name!='z') && (i_vector_name!='Z') )
          {
             fprintf(stderr,"VMAXPS in vec_compute_reg expects zmm registers\n");
             exit(-1);
          }
          l_fpadj2 = -0x81;
          l_fpadj = 6;
          break;
       case LIBXSMM_X86_INSTR_VCVTDQ2PS:
          l_fpadj2 -= 0x81;
          l_fpadj += 0x02;
          if ( l_vreg2 != LIBXSMM_X86_VEC_REG_UNDEF )
          {
             LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VEC_REG_MUST_BE_UNDEF );
          }
          l_vreg2 = l_vreg1;
          l_vreg1 = 0;
          break;
       case LIBXSMM_X86_INSTR_VCVTPS2PD:
          l_fpadj2 -= 0x81;
          l_fpadj += 0x01;
          if ( l_vreg2 != LIBXSMM_X86_VEC_REG_UNDEF )
          {
             LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VEC_REG_MUST_BE_UNDEF );
/*
             fprintf(stderr,"Please call VCVTPS2PD with regs 0 and 1. Use UNDEF with reg 2\n");
*/
             exit(-1);
          }
          l_vreg2 = l_vreg1;
          l_vreg1 = 0;
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
          } else if ( l_vreg0 > 7 ) {
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
          } else if ( l_vreg0 > 7 ) {
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
          } else if ( l_vreg0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VMULSD:
          l_fpadj2 = 2;
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VMULSD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VADDSD:
          l_fpadj  =-1;
          l_fpadj2 = 2;
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VADDSD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VSUBSD:
          l_fpadj  = 3;
          l_fpadj2 = 2;
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VSUBSD and ymm/zmm?\n");
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SD:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: Really? VFMADD231SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x60;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( l_vreg0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231SD:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VFMSUB231SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x62;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( l_vreg0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231SD:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VFNMADD231SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x64;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( l_vreg0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231SD:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VFNMSUB231SD and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x66;
          l_fpadj2 += 0x80;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( l_vreg0 > 7 ) {
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
          if ( (i_vector_name!='z') && (l_vreg0<=15) &&
               (l_vreg1<=15) && (l_vreg2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          break;
       case LIBXSMM_X86_INSTR_VADDPS:
          if ( (i_vector_name!='z') && (l_vreg0<=15) &&
               (l_vreg1<=15) && (l_vreg2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = -1;
          break;
       case LIBXSMM_X86_INSTR_VSUBPS:
          if ( (i_vector_name!='z') && (l_vreg0<=15) &&
               (l_vreg1<=15) && (l_vreg2<=15) )
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
          } else if ( l_vreg0 > 7 ) {
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
          } else if ( l_vreg0 > 7 ) {
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
          } else if ( l_vreg0 > 7 ) {
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
          } else if ( l_vreg0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VPSRAVD:
          l_second += 0x01;
          l_fpadj  -= 0x13;
          l_fpadj2 -= 0x80;
          break;
       /* SSE instruction support */
       case LIBXSMM_X86_INSTR_VMULSS:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VMULSS and ymm/zmm?\n");
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VADDSS:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VADDSS and ymm/zmm?\n");
          l_fpadj  =-1;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VSUBSS:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VSUBSS and ymm/zmm?\n");
          l_fpadj  = 3;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SS:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VFMADD231SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x60;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( l_vreg0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFMSUB231SS:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VFMSUB231SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x62;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( l_vreg0 > 7 ) {
             l_second -= 0x20;
          }
          l_bytes = 5;
          break;
       case LIBXSMM_X86_INSTR_VFNMADD231SS:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VFNMADD231SS and ymm/zmm?\n");
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
       case LIBXSMM_X86_INSTR_VPERMT2W:
          l_second += 0x01;
          l_fpadj  += 0x24;
          if ( i_vector_name == 'x' )
          {
             l_fourth -= 0x40;
             if ( l_vreg0 >= 16 ) l_fourth -= 0xc0;
             if ( l_vreg1 >= 16 ) l_fourth -= 0xc0;
          } else if ( i_vector_name == 'y' )
          {
             l_fourth -= 0x20;
             if ( l_vreg0 >= 16 ) l_fourth += 0x20;
             if ( l_vreg1 >= 16 ) l_fourth += 0x20;
          }
          l_bytes = 6;
          break;
       case LIBXSMM_X86_INSTR_VFNMSUB231SS:
          if (i_vector_name != 'x') fprintf(stderr, "libxsmm_instruction_vec_compute_reg: VFNMSUB231SS and ymm/zmm?\n");
          l_second += 0x21;
          l_fpadj  += 0x66;
          if ( i_vector_name == 'z' )
          {
             l_second -= 0x20;
             l_fpadj2 -= 0x80;
          } else if ( l_vreg0 > 7 ) {
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
       case LIBXSMM_X86_INSTR_VPORD:
          l_bytes = 6;
          if ( i_vector_name == 'x' )
          {
             l_fourth -= 0x40;
          } else if ( i_vector_name == 'y' )
          {
             l_fourth -= 0x20;
          }
          l_fpadj += 0x92;
          l_fpadj2 += 0x80;
          break;
        case LIBXSMM_X86_INSTR_VPDPWSSD:
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) &&
               (i_vec_reg_number_1<=15) && (i_vec_reg_number_2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = -0x07;
          l_second += 0x01;
          l_third += 0x01;
          break;
        case LIBXSMM_X86_INSTR_VPDPWSSDS:
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) &&
               (i_vec_reg_number_1<=15) && (i_vec_reg_number_2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = -0x06;
          l_second += 0x01;
          l_third += 0x01;
          break;
        case LIBXSMM_X86_INSTR_VPDPBUSD:
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) &&
               (i_vec_reg_number_1<=15) && (i_vec_reg_number_2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = -0x09;
          l_second += 0x01;
          l_third += 0x01;
          break;
        case LIBXSMM_X86_INSTR_VPDPBUSDS:
          if ( (i_vector_name!='z') && (i_vec_reg_number_0<=15) &&
               (i_vec_reg_number_1<=15) && (i_vec_reg_number_2<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = -0x08;
          l_second += 0x01;
          l_third += 0x01;
          break;
       case LIBXSMM_X86_INSTR_MOVAPD:
          l_sse = 1;
          l_insert_extra_byte = 0x66;
          l_third = 0x18;
          break;
       case LIBXSMM_X86_INSTR_MOVUPD:
          l_sse = 1;
          l_insert_extra_byte = 0x66;
          break;
       case LIBXSMM_X86_INSTR_MOVAPS:
          l_sse = 1;
          l_third = 0x18;
          break;
       case LIBXSMM_X86_INSTR_MOVUPS:
          l_sse = 1;
          break;
       case LIBXSMM_X86_INSTR_MOVSD:
          l_sse = 1;
          l_insert_extra_byte = 0xF2;
          break;
       case LIBXSMM_X86_INSTR_MOVSS:
          l_sse = 1;
          l_insert_extra_byte = 0xF3;
          break;
       case LIBXSMM_X86_INSTR_MOVDDUP:
          l_sse = 1;
          l_third = 2;
          l_insert_extra_byte = 0xF2;
          break;
       case LIBXSMM_X86_INSTR_XORPD:
          l_sse = 1;
          l_insert_extra_byte = 0x66;
          l_third = 0x47;
          break;
       case LIBXSMM_X86_INSTR_XORPS:
          l_sse = 1;
          l_third = 0x47;
          break;
       case LIBXSMM_X86_INSTR_MULPD:
          l_sse = 1;
          l_insert_extra_byte = 0x66;
          l_third = 0x49;
          break;
       case LIBXSMM_X86_INSTR_MULPS:
          l_sse = 1;
          l_third = 0x49;
          break;
       case LIBXSMM_X86_INSTR_ADDPD:
          l_sse = 1;
          l_insert_extra_byte = 0x66;
          l_third = 0x48;
          break;
       case LIBXSMM_X86_INSTR_ADDPS:
          l_sse = 1;
          l_third = 0x48;
          break;
       case LIBXSMM_X86_INSTR_SUBPD:
          l_sse = 1;
          l_insert_extra_byte = 0x66;
          l_third = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_SUBPS:
          l_sse = 1;
          l_third = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_MULSD:
          l_sse = 1;
          l_insert_extra_byte = 0xF2;
          l_third = 0x49;
          break;
       case LIBXSMM_X86_INSTR_MULSS:
          l_sse = 1;
          l_insert_extra_byte = 0xF3;
          l_third = 0x49;
          break;
       case LIBXSMM_X86_INSTR_ADDSD:
          l_sse = 1;
          l_insert_extra_byte = 0xF2;
          l_third = 0x48;
          break;
       case LIBXSMM_X86_INSTR_ADDSS:
          l_sse = 1;
          l_insert_extra_byte = 0xF3;
          l_third = 0x48;
          break;
       case LIBXSMM_X86_INSTR_SUBSD:
          l_sse = 1;
          l_insert_extra_byte = 0xF2;
          l_third = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_SUBSS:
          l_sse = 1;
          l_insert_extra_byte = 0xF3;
          l_third = 0x4c;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_reg: Unknown instruction type: %u\n", i_vec_instr);
          exit(-1);
    }
    l_reg0 = l_vreg0 % 8;
    l_reg1 = l_vreg1 % 8;
    l_reg2 = l_vreg2 % 8;
    if ( !l_sse )
    {
       if ( i_vector_name == 'x' ) l_xreg = -4;
       if ( l_vreg2 >= 8 ) { l_second -= 0x80; }
       if ( l_vreg1 >= 8 ) { l_third  -= 0x40; }
       if ( (i_vector_name!='z') && (l_vreg0<=15) &&
            (l_vreg1<=15) && (l_vreg2<=15) )
       {
          if ( l_vreg0 >= 8 )
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
          if ( l_vreg0 >= 8 ) { l_second -= 0x20; }
          if ( l_vreg0 >= 16 )
          {
             l_second -= 0x20;
             if ( i_vector_name=='x' ) l_fourth -= 0x40;
             if ( i_vector_name=='y' ) l_fourth -= 0x20;
          }
          if ( l_vreg0 >= 24 ) { l_second -= 0x20; }
          if ( l_vreg1 >= 16 )
          {
             l_third += 0x40;
             l_fourth -= 0x08;
             if ( i_vector_name=='x' ) l_fourth -= 0x40;
             if ( i_vector_name=='y' ) l_fourth -= 0x20;
          }
          if ( l_vreg1 >= 24 ) { l_third -= 0x40; }
          if ( l_vreg2 >= 16 ) { l_second += 0x70; }
          if ( l_vreg2 >= 24 ) { l_second -= 0x80; }
          buf[i++] = 0x62;
          buf[i++] = (unsigned char)(0xf1 + l_second);
          buf[i++] = (unsigned char)(0xfd - 8*l_reg1   + l_third + l_fpadj2);
          buf[i++] = (unsigned char)(0x48 + l_fourth);
          buf[i++] = (unsigned char)(0x59 + l_fpadj);
          buf[i++] = (unsigned char)(0xc0 + l_reg0    + 8*l_reg2);
       }
    } else {
       int l_vecgrp0 = 0;
       int l_vecgrp1 = 0;
       if ( (l_vreg0 >= 8) && (l_vreg0 <=15) ) l_vecgrp0 = 1;
       if ( (l_vreg1 >= 8) && (l_vreg1 <=15) ) l_vecgrp1 = 1;
       if ( l_insert_extra_byte != 0 )
       {
          buf[i++] = (unsigned char)(l_insert_extra_byte);
       }
       if ( (l_vecgrp0 >= 1) || (l_vecgrp1 >= 1) )     {
          int l_extra_byte = 0;
          if ( l_vecgrp0 >= 1 ) l_extra_byte += 1;
          if ( l_vecgrp1 >= 1 ) l_extra_byte += 4;
          buf[i++] = (unsigned char)(0x40 + l_extra_byte);
       }
       buf[i++] = (unsigned char)(0x0f);
       buf[i++] = (unsigned char)(0x10 + l_third);
       buf[i++] = (unsigned char)(0xc0 + l_reg0 + l_reg1*8);
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

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_reg_mask( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_instruction_set,
                                                   const unsigned int      i_vec_instr,
                                                   const char              i_vector_name,
                                                   const unsigned int      i_vec_reg_number_0,
                                                   const unsigned int      i_vec_reg_number_1,
                                                   const unsigned int      i_vec_reg_number_2,
                                                   const unsigned int      i_immediate,
                                                   const unsigned int      i_mask_reg_number,
                                                   const unsigned int      i_use_zero_masking )
{
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /* unsigned int l_maxsize = 1024; */
    int l_vecval0 = i_vec_reg_number_0 % 8;
    int l_vecgrp0 = i_vec_reg_number_0 / 8;
    int l_oddgrp0 = ((l_vecgrp0 % 2)==1);
    int l_2or3grp0 = (l_vecgrp0>=2);
    int l_vecval1 = i_vec_reg_number_1 % 8;
    int l_vecgrp1 = i_vec_reg_number_1 / 8;
    int l_oddgrp1 = ((l_vecgrp1 % 2)==1);
    int l_2or3grp1 = (l_vecgrp1>=2);
    int l_vecval2 = i_vec_reg_number_2 % 8;
    int l_vecgrp2 = i_vec_reg_number_2 / 8;
    int l_oddgrp2 = ((l_vecgrp2 % 2)==1);
    int l_2or3grp2 = (l_vecgrp2>=2);
    int l_second = 0;
    int l_third = 0;
    int l_fourth = 0;
    int l_fifth = 0;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }

    switch ( i_vector_name ) {
       case 'x':
       case 'y':
          fprintf(stderr, "libxsmm_instruction_vec_compute_reg_mask: the highest register should be zmm: use that\n");
          exit(-1);
          break;
       case 'z':
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_reg_mask: Unknown sort of fp registers\n");
          exit(-1);
    }

    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_VBLENDMPS:
          if ( i_immediate != LIBXSMM_X86_IMM_UNDEF )
          {
              fprintf(stderr,"libxsmm_instruction_vec_compute_reg_mask immediate=%u != %i\n",i_immediate,LIBXSMM_X86_IMM_UNDEF);
              exit(-1);
          }
          l_second = 0x1;
          l_fourth = i_mask_reg_number;
          break;
       case LIBXSMM_X86_INSTR_VPCMPD:
          l_second = 0x2;
          l_fifth = -0x46;
          l_oddgrp2 = 0;
          l_2or3grp2 = 0;
          l_vecval2 = i_mask_reg_number;
          break;
       case LIBXSMM_X86_INSTR_VCMPPS:
          l_third = -1;
          l_fifth = 0x5d;
          l_oddgrp2 = 0;
          l_2or3grp2 = 0;
          l_vecval2 = i_mask_reg_number;
          break;
       case LIBXSMM_X86_INSTR_VPADDD:
          if ( i_immediate != LIBXSMM_X86_IMM_UNDEF )
          {
              fprintf(stderr,"libxsmm_instruction_vec_compute_reg_mask immediate=%u != %i\n",i_immediate,LIBXSMM_X86_IMM_UNDEF);
              exit(-1);
          }
          l_fifth = 0x99;
          l_fourth = i_mask_reg_number;
          break;
       case LIBXSMM_X86_INSTR_VPANDD:
          if ( i_immediate != LIBXSMM_X86_IMM_UNDEF )
          {
              fprintf(stderr,"libxsmm_instruction_vec_compute_reg_mask immediate=%u != %i\n",i_immediate,LIBXSMM_X86_IMM_UNDEF);
              exit(-1);
          }
          l_fifth = 0x76;
          l_fourth = i_mask_reg_number;
          break;
       case LIBXSMM_X86_INSTR_VPSUBD:
          if ( i_immediate != LIBXSMM_X86_IMM_UNDEF )
          {
              fprintf(stderr,"libxsmm_instruction_vec_compute_reg_mask immediate=%u != %i\n",i_immediate,LIBXSMM_X86_IMM_UNDEF);
              exit(-1);
          }
          l_fifth = 0x95;
          l_fourth = i_mask_reg_number;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_reg_mask: Unknown instruction type: %u\n", i_vec_instr);
          exit(-1);
    }

    if ( i_use_zero_masking != 0 && i_mask_reg_number != 0 ) l_fourth += 0x80;

    buf[i++] = (unsigned char)(0x62);
    buf[i++] = (unsigned char)(0xf1 + l_second - l_oddgrp0 * 0x20 - l_oddgrp2 * 0x80 - l_2or3grp0 * 0x40 - l_2or3grp2 * 0x10);
    buf[i++] = (unsigned char)(0x7d + l_third - l_oddgrp1 * 0x40 - l_vecval1*8);
    buf[i++] = (unsigned char)(0x48 - l_2or3grp1 * 0x08 + l_fourth );
    buf[i++] = (unsigned char)(0x65 + l_fifth);
    buf[i++] = (unsigned char)(0xc0 + l_vecval0 + l_vecval2*8);

    if ( i_immediate != LIBXSMM_X86_IMM_UNDEF ) {
       buf[i++] = (unsigned char)(i_immediate);
    }

    io_generated_code->code_size = i;
    /* *loc = i; */

  } else {
    /* TODO: Debug- this code was just copied from another routine */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[16];
    char l_masking[16];

    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name, 15 );

    if ( i_mask_reg_number != 0 ) {
      /* avoid format-truncation warning due to unsigned int (theoretically) exceeding length of string (l_masking) */
      LIBXSMM_ASSERT_MSG(i_mask_reg_number < 8, "Invalid mask register");
      if ( i_use_zero_masking == 0 ) {
        if ( io_generated_code->code_type == 0 ) {
          LIBXSMM_SNPRINTF(l_masking, 16, "%%{k%hd%%}", (unsigned short)i_mask_reg_number);
        } else {
          LIBXSMM_SNPRINTF(l_masking, 16, "{k%hd}", (unsigned short)i_mask_reg_number);
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          LIBXSMM_SNPRINTF(l_masking, 16, "%%{k%hd%%}%%{z%%}", (unsigned short)i_mask_reg_number);
        } else {
          LIBXSMM_SNPRINTF(l_masking, 16, "{k%hd}{z}", (unsigned short)i_mask_reg_number);
        }
      }
    }
    else l_masking[0] = (char)0; /* no mask */

    /* build vXYZpd/ps/sd/ss instruction pure register use*/
    if ( i_instruction_set >= LIBXSMM_X86_AVX512 ) {
      if ( io_generated_code->code_type == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %%%%%cmm%u, %%%%%cmm%u%s\\n\\t\"\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2, l_masking );
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %%%cmm%u, %%%cmm%u%s\n", l_instr_name, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, i_vector_name, i_vec_reg_number_2, l_masking );
      }
    } else {
      /* This is an error */
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_API_INTERN
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
  if ( (i_instruction_set < LIBXSMM_X86_AVX512)  &&
       (i_use_broadcast != 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_NO_AVX512_BCAST );
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
    int l_vec_0 = i_vec_reg_number_0;
    int l_vec_1 = i_vec_reg_number_1;
    int l_reg1   = l_vec_0;
    int l_reg2   = l_vec_1;
    int l_fpadj=0, l_place=0;
    int l_fpadj2=0;
    int l_bytes=4;
    int l_regi=0;
    int l_forced_offset=0;
    int l_sizereg=64;
    int l_scaleadj=0;
    int l_sse3 = 0;
    int l_insert_extra_byte = 0;
    /* int l_iregoff = 0; */

    int l_broadcast = (int)i_use_broadcast;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }
    switch ( i_vector_name ) {
       case 'x':
          l_sizereg = 1;
          if ( l_broadcast != 0 )
          {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: broadcasts aren't enabled with xmm yet\n");
             exit(-1);
          }
          break;
       case 'y':
          l_sizereg = 1;
          if ( l_broadcast != 0 )
          {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: broadcasts aren't enabled with ymm yet\n");
             exit(-1);
          }
          break;
       case 'z':
          l_bytes = 6;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_mem: Unknown sort of fp registers\n");
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
       case LIBXSMM_X86_INSTR_VPANDD:
          l_fpadj = 0x82;
          l_fpadj2 = 0x80;
          if ( l_broadcast != 0 ) l_sizereg = 4;
          break;
       case LIBXSMM_X86_INSTR_VSUBPD:
          l_fpadj = 3;
          break;
       case LIBXSMM_X86_INSTR_VMAXPD:
          l_fpadj = 6;
          break;
       case LIBXSMM_X86_INSTR_VMAXPS:
          l_fpadj = 6;
          l_fpadj2 = -0x81;
          break;
       case LIBXSMM_X86_INSTR_VPERMW:
          l_second += 0x01;
          l_fpadj = 0x34;
          break;
       case LIBXSMM_X86_INSTR_VPERMD:
          l_second += 0x01;
          l_fpadj = -0x23;
          l_fpadj2 = -0x80;
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
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vmulsd and ymm/zmm?\n");
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_VADDSD:
          l_fpadj  =-1;
          l_fpadj2 = 2;
          if (i_vector_name != 'x')
          {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vaddsd and ymm/zmm?\n");
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_VSUBSD:
          l_fpadj  = 3;
          l_fpadj2 = 2;
          if (i_vector_name != 'x')
          {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vsubsd and ymm/zmm?\n");
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SD:
          if (i_vector_name != 'x')
          {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vfmadd231sd and ymm/zmm?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vfmsub231sd and ymm/zmm?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vfnmadd231sd and ymm/zmm?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vfnmsub231sd and ymm/zmm?\n");
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
       case LIBXSMM_X86_INSTR_VPMOVDW:
          l_bytes = 6;
          l_second += 0x1;
          l_fpadj2 -= 0x7F;
          l_fpadj -= 0x26;
          l_sizereg = 32;
          break;
       case LIBXSMM_X86_INSTR_VPMOVSXWD:
          l_bytes = 5;
          l_second += 0x21;
          l_fpadj -= 0x36;
          l_fpadj2 -= 0x0;
          l_sizereg = 1;
          break;
       case LIBXSMM_X86_INSTR_VPMOVZXWD:
          l_bytes = 5;
          l_second += 0x21;
          l_fpadj -= 0x26;
          l_fpadj2 -= 0x0;
          l_sizereg = 1;
          break;
       case LIBXSMM_X86_INSTR_VPMOVSXBD:
          l_bytes = 5;
          l_second += 0x21;
          l_fpadj -= 0x38;
          l_fpadj2 -= 0x0;
          l_sizereg = 1;
          break;
       case LIBXSMM_X86_INSTR_VPMOVZXBD:
          l_bytes = 5;
          l_second += 0x21;
          l_fpadj -= 0x28;
          l_fpadj2 -= 0x0;
          l_sizereg = 1;
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
       case LIBXSMM_X86_INSTR_VDPBF16PS:
          if ( i_vector_name=='y' ) { l_sizereg = 32; l_fourth -= 0x20; }
          if ( i_vector_name=='x' ) { l_sizereg = 16; l_fourth -= 0x40; }
          if ( l_broadcast == 1 ) l_sizereg = 4;
#if !defined(NDEBUG) /* TODO: code protected by !defined(NDEBUG) is identical in both branches */
          if ( (i_vector_name!='z') && (l_vec_0<=15) && (l_vec_1<=15) ) {
               l_fpadj2 = -0x81;
          }
          else
#endif
          {
               l_fpadj2 = -0x81;
          }
          l_fpadj2 += 0x02;
          l_fpadj = -7;
          l_second += 1;
          l_bytes = 6;
          break;
       case LIBXSMM_X86_INSTR_VADDPS:
          if ( l_broadcast == 1 ) l_sizereg = 4;
          if ( (i_vector_name!='z') && (l_vec_0<=15) && (l_vec_1<=15) )
               l_fpadj2 = -1;
          else l_fpadj2 = -0x81;
          l_fpadj = -1;
          break;
       case LIBXSMM_X86_INSTR_VSUBPS:
          if ( l_broadcast == 1 ) l_sizereg = 4;
          if ( (i_vector_name!='z') && (l_vec_0<=15) && (l_vec_1<=15) )
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
       case LIBXSMM_X86_INSTR_VFMADD213PS:
          if ( l_broadcast == 1 ) l_sizereg = 4;
          l_second += 0x21;
          l_fpadj  += 0x4f;
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
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vmulss and ymm/zmm?\n");
             exit(-1);
          }
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VADDSS:
          if (i_vector_name != 'x')
          {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vaddss and ymm/zmm?\n");
             exit(-1);
          }
          l_fpadj  =-1;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VSUBSS:
          if (i_vector_name != 'x')
          {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vsubss and ymm/zmm?\n");
             exit(-1);
          }
          l_fpadj  = 3;
          l_fpadj2 = 1;
          break;
       case LIBXSMM_X86_INSTR_VFMADD231SS:
          if (i_vector_name != 'x')
          {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vfmadd231ss and ymm/zmm?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vfmsub231ss and ymm/zmm?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vfnmadd231ss and ymm/zmm?\n");
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
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vfnmsub231ss and ymm/zmm?\n");
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
       case LIBXSMM_X86_INSTR_VPORD:
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
          l_fpadj += 0x92;
          l_fpadj2 += 0x80;
          break;
       case LIBXSMM_X86_INSTR_VPSRAVD:
          l_second += 0x01;
          l_fpadj  -= 0x13;
          l_fpadj2 -= 0x80;
          if ( l_broadcast == 1 ) l_sizereg = 4;
          break;
       case LIBXSMM_X86_INSTR_VPADDD:
          l_fpadj2 -= 0x80;
          l_fpadj  += 0xA5;
          if ( l_broadcast == 1 ) l_sizereg = 4;
          break;
       case LIBXSMM_X86_INSTR_VPDPWSSD:
          l_second += 0x01;
          l_fpadj  -= 0x07;
          l_fpadj2 -= 0x80;
          if ( l_broadcast == 1 ) l_sizereg = 4;
          if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_RSP ) {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vpdpwssd and idx=rsp?\n");
             exit(-1);
          }
          break;
        case LIBXSMM_X86_INSTR_VPDPWSSDS:
          l_second += 0x01;
          l_fpadj  -= 0x06;
          l_fpadj2 -= 0x80;
          if ( l_broadcast == 1 ) l_sizereg = 4;
          if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_RSP ) {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vpdpwssds and idx=rsp?\n");
             exit(-1);
          }
          break;
        case LIBXSMM_X86_INSTR_VPDPBUSD:
          l_second += 0x01;
          l_fpadj  -= 0x09;
          l_fpadj2 -= 0x80;
          if ( l_broadcast == 1 ) l_sizereg = 4;
          if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_RSP ) {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vpdpbusd and idx=rsp?\n");
             exit(-1);
          }
          break;
        case LIBXSMM_X86_INSTR_VPDPBUSDS:
          l_second += 0x01;
          l_fpadj  -= 0x08;
          l_fpadj2 -= 0x80;
          if ( l_broadcast == 1 ) l_sizereg = 4;
          if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_RSP ) {
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem: vpdpbusds and idx=rsp?\n");
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_XORPD:
          l_sse3 = 1;
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x47;
          break;
       case LIBXSMM_X86_INSTR_XORPS:
          l_sse3 = 1;
          l_fpadj = 0x47;
          break;
       case LIBXSMM_X86_INSTR_MULPD:
          l_sse3 = 1;
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x49;
          break;
       case LIBXSMM_X86_INSTR_MULSS:
          l_sse3 = 1;
          l_insert_extra_byte = 0xF3;
          l_fpadj = 0x49;
          break;
       case LIBXSMM_X86_INSTR_MULPS:
          l_sse3 = 1;
          l_fpadj = 0x49;
          break;
       case LIBXSMM_X86_INSTR_ADDPD:
          l_sse3 = 1;
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x48;
          break;
       case LIBXSMM_X86_INSTR_ADDSS:
          l_sse3 = 1;
          l_insert_extra_byte = 0xF3;
          l_fpadj = 0x48;
          break;
       case LIBXSMM_X86_INSTR_ADDPS:
          l_sse3 = 1;
          l_fpadj = 0x48;
          break;
       case LIBXSMM_X86_INSTR_ADDSD:
          l_sse3 = 1;
          l_insert_extra_byte = 0xF2;
          l_fpadj = 0x48;
          break;
       case LIBXSMM_X86_INSTR_SUBPD:
          l_sse3 = 1;
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_SUBSS:
          l_sse3 = 1;
          l_insert_extra_byte = 0xF3;
          l_fpadj = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_SUBPS:
          l_sse3 = 1;
          l_fpadj = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_SUBSD:
          l_sse3 = 1;
          l_insert_extra_byte = 0xF2;
          l_fpadj = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_MULSD:
          l_sse3 = 1;
          l_insert_extra_byte = 0xF2;
          l_fpadj = 0x49;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_mem: Unknown instruction type: %u\n", i_vec_instr);
          exit(-1);
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
            fprintf(stderr, "libxsmm_instruction_vec_compute_mem: cannot handle i_scale=%u parameter\n", i_scale);
            exit(-1);
       }
    }
    if ( !l_sse3 )
    {
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
    l_reg0 = i_gp_reg_base % 8;
    l_reg1 = l_vec_0 % 8;
    l_reg2 = l_vec_1 % 8;
    if ( i_vec_instr == LIBXSMM_X86_INSTR_VPMOVDW )
    {
       /* We only have 1 vector register input */
       l_reg2 = l_vec_0 % 8;
       l_reg1 = 0;
       l_vec_1 = l_vec_0;
       l_vec_0 = 0;
    }
    if ( (i_vec_instr == LIBXSMM_X86_INSTR_VPMOVSXWD) ||
         (i_vec_instr == LIBXSMM_X86_INSTR_VPMOVZXWD) ||
         (i_vec_instr == LIBXSMM_X86_INSTR_VPMOVSXBD) ||
         (i_vec_instr == LIBXSMM_X86_INSTR_VPMOVZXBD) )
    {
       /* We only have 1 vector register input */
       l_reg2 = i_vec_reg_number_0 % 8;
       l_reg1 = 0;
       l_vec_0 = 0;
       l_vec_1 = i_vec_reg_number_0;
       if ((i_gp_reg_base >= 8) && (i_gp_reg_base != LIBXSMM_X86_GP_REG_UNDEF))
       {
          if ((i_gp_reg_idx < 8) && (i_gp_reg_idx  != LIBXSMM_X86_GP_REG_UNDEF))
          {
             l_second -= 0x20;
          }
       }
       if ((i_gp_reg_base < 8) && (i_gp_reg_base != LIBXSMM_X86_GP_REG_UNDEF))
       {
          if ((i_gp_reg_idx >= 8) && (i_gp_reg_idx  != LIBXSMM_X86_GP_REG_UNDEF))
          {
             l_second -= 0x20;
          }
          if ( i_vector_name == 'z' )
          {
             if ( (i_gp_reg_idx < 8) || (i_gp_reg_idx==LIBXSMM_X86_GP_REG_UNDEF) )
             {
                l_second += 0xE0;
             }
             if ( (i_vec_instr == LIBXSMM_X86_INSTR_VPMOVSXBD) ||
                  (i_vec_instr == LIBXSMM_X86_INSTR_VPMOVZXBD) )
             {
                l_sizereg = 16;
             } else {
                l_sizereg = 32;
             }
          }
       }
       if ( i_vector_name == 'z' ) l_third -= 0x80;
    }
    if ( l_vec_0 >= 8 ) { l_third  -= 0x40; }
    if ( l_vec_1 >= 8 ) { l_second -= 0x80; }
    if ( (i_vector_name == 'z') || (l_vec_0 > 15) || (l_vec_1 > 15) )
        l_bytes = 6;

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

/*     if ( l_vec_0 >= 8 ) { l_third -= 0x40; } */
       if ( l_vec_0 >= 16) { l_third += 0x40; l_fourth -= 0x8; }
       if ( l_vec_0 >= 24) { l_third -= 0x40; }

/*     if ( l_vec_1 >= 8 ) { l_second -= 0x80; } */
       if ( l_vec_1 >= 16) { l_second += 0x70; }
       if ( l_vec_1 >= 24) { l_second -= 0x80; }
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

    i += internal_x86_instructions_add_offset( l_place, i, i_displacement, l_forced_offset, l_sizereg, buf );

    io_generated_code->code_size = i;
    /* *loc = i; */
    } else { /* SSE3 code */
        int l_vecgrp0 = 0;
        int l_vecval0 = i_vec_reg_number_0 % 8;
        int l_place1=i+2;
        int l_regbas0 = i_gp_reg_base % 8;
        int l_regidx =  i_gp_reg_idx % 8;
        int l_gp8 = ((i_gp_reg_base > 7)&&(i_gp_reg_base<=15)?1:0);
        LIBXSMM_ASSERT(0 == l_forced_offset);
        if ( (i_vec_reg_number_0>=8) && (i_vec_reg_number_0<=15) ) l_vecgrp0=1;
        if ( l_insert_extra_byte != 0 )
        {
            buf[i++]= (unsigned char)(l_insert_extra_byte);
            ++l_place1;
        }
        if (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF )
        {
            int l_sse_preamble2 = 64;
            if ( l_gp8 || (l_vecgrp0>=1) )
            {
               if (l_gp8) l_sse_preamble2 += 1;
               if (l_vecgrp0 >=1) l_sse_preamble2 += 4;
               buf[i++] = (unsigned char)(l_sse_preamble2);
               ++l_place1;
            }
            buf[i++] = (unsigned char)(0x0f);
            buf[i++] = (unsigned char)(0x10 + l_fpadj);
            buf[i++] = (unsigned char)(0x00 + l_regbas0 + l_vecval0*8);
            if ( l_regbas0 == 4 ) buf[i++]=0x24;
        } else {
            int l_sse_preamble2 = 64;
            int l_ix8 = ((i_gp_reg_idx > 7)&&(i_gp_reg_idx<=15)?1:0);
            if ( l_gp8 || l_ix8 || (l_vecgrp0>=1) )
            {
                if (l_gp8) l_sse_preamble2 += 1;
                if (l_ix8) l_sse_preamble2 += 2;
                if (l_vecgrp0 >=1) l_sse_preamble2 += 4;
                buf[i++] = (unsigned char)(l_sse_preamble2);
                ++l_place1;
            }
            buf[i++] = (unsigned char)(0x0f);
            buf[i++] = (unsigned char)(0x10 + l_fpadj);
            buf[i++] = (unsigned char)(0x04 + l_vecval0*8);
            buf[i++] = (unsigned char)(0x00 + l_scaleadj + l_regbas0 + l_regidx*8);
        }
        if ( (l_regbas0 == 5) && (i_displacement==0) )
        {
            l_forced_offset = 1;
        }
        i += internal_x86_instructions_add_offset( l_place1, i, i_displacement, l_forced_offset, 1, buf );
        /* *loc = i; */
        io_generated_code->code_size = i;
     }
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
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s) {%s}, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
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
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u) {%s}, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u), %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
        }
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_mask ( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_instruction_set,
                                                    const unsigned int      i_vec_instr,
                                                    const unsigned int      i_use_broadcast,
                                                    const unsigned int      i_gp_reg_base,
                                                    const unsigned int      i_gp_reg_idx,
                                                    const unsigned int      i_scale,
                                                    const int               i_displacement,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_vec_reg_number_0,
                                                    const unsigned int      i_vec_reg_number_1,
                                                    const unsigned int      i_immediate,
                                                    const unsigned int      i_mask_reg_number,
                                                    const unsigned int      i_use_zero_masking )
{
  /* @TODO add checks in debug mode */
  if ( (i_instruction_set < LIBXSMM_X86_AVX512)  &&
       (i_use_broadcast != 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_NO_AVX512_BCAST );
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /*int i = *loc;*/
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /*unsigned int l_maxsize = 1024;*/

    int l_regbas0 = i_gp_reg_base % 8;
    int l_gp8     = ((i_gp_reg_base > 7)&&(i_gp_reg_base<=15)?1:0);
    int l_regidx  = i_gp_reg_idx  % 8;
    int l_ix8     = ((i_gp_reg_idx > 7)&&(i_gp_reg_idx<=15)?1:0);
    int l_vecval0 = i_vec_reg_number_0 % 8;
    int l_vecgrp0 = i_vec_reg_number_0 / 8;
    int l_oddgrp0 = ((l_vecgrp0 % 2)==1);
    int l_2or3grp0 = (l_vecgrp0>=2);
    int l_vecval1 = i_vec_reg_number_1 % 8;
    int l_vecgrp1 = i_vec_reg_number_1 / 8;
    int l_oddgrp1 = ((l_vecgrp1 % 2)==1);
    int l_2or3grp1 = (l_vecgrp1>=2);
    int l_scaleadj = 0;
    int l_place = i;
    int l_sizereg = 64;
    int l_forced_offset = 0;
    int l_second = 0;
    int l_third = 0;
    int l_fourth = 0;
    int l_fifth = 0;
    int l_sixth = 0;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }
    if ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_UNDEF) ||
         (((int)i_gp_reg_base < LIBXSMM_X86_GP_REG_RAX) || (i_gp_reg_base > LIBXSMM_X86_GP_REG_R15)) )
    {
       fprintf(stderr,"libxsmm_instruction_vec_compute_mem_mask has invalid i_gp_reg_base input\n");
       exit(-1);
    }
    if ( (i_gp_reg_idx  != LIBXSMM_X86_GP_REG_UNDEF) &&
         (((int)i_gp_reg_idx < LIBXSMM_X86_GP_REG_RAX) || (i_gp_reg_idx > LIBXSMM_X86_GP_REG_R15)) )
    {
       fprintf(stderr,"libxsmm_instruction_vec_compute_mem_mask has invalid i_gp_reg_idx input\n");
       exit(-1);
    }

    switch ( i_vector_name ) {
       case 'x':
       case 'y':
          fprintf(stderr, "libxsmm_instruction_vec_compute_mem_mask: xmm/ymm not enabled yet\n");
          exit(-1);
          break;
       case 'z':
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_mem_mask: Unknown sort of fp registers\n");
          exit(-1);
    }

    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_VCMPPS:
          l_place = i + 5;
          l_sizereg = 64;
          l_fifth = 0xA3;
          l_sixth = i_mask_reg_number*8;
          l_vecval1 = 0;
          l_vecgrp1 = 0;
          l_oddgrp1 = 0;
          l_2or3grp1 = 0;
          break;
       case LIBXSMM_X86_INSTR_VPCMPD:
          l_place = i + 5;
          l_sizereg = 64;
          l_second = 2;
          l_third = 1;
          l_sixth = i_mask_reg_number*8;
          l_vecval1 = 0;
          l_vecgrp1 = 0;
          l_oddgrp1 = 0;
          l_2or3grp1 = 0;
          break;
       case LIBXSMM_X86_INSTR_VPADDD:
          if ( i_immediate != LIBXSMM_X86_IMM_UNDEF ) {
             fprintf(stderr,"libxsmm_instruction_vec_compute_mem_mask: vpaddd should not use an immediate. You passed %u not %i\n",i_immediate,LIBXSMM_X86_IMM_UNDEF);
             exit(-1);
          }
          l_place = i + 5;
          l_sizereg = 64;
          l_third = 1;
          l_fourth = i_mask_reg_number;
          l_fifth = 0xDF;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_mem_mask: Unknown instruction type: %u\n", i_vec_instr);
          exit(-1);
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
             fprintf(stderr, "libxsmm_instruction_vec_compute_mem_mask: cannot handle i_scale=%u parameter\n", i_scale);
             exit(-1);
       }
    }

    if ( i_use_broadcast ) { l_fourth += 0x10; l_sizereg = 4; }
    if ( i_use_zero_masking != 0 && i_mask_reg_number != 0 ) l_fourth += 0x80;

    if (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF )
    {
        buf[i++] = (unsigned char)(0x62);
        buf[i++] = (unsigned char)(0xf1 + l_second - l_gp8 * 0x20 - l_oddgrp1 * 0x80 - l_2or3grp1 * 0x10 );
        buf[i++] = (unsigned char)(0x7c + l_third - l_oddgrp0 * 0x40 - l_vecval0*8);
        buf[i++] = (unsigned char)(0x48 + l_fourth - l_2or3grp0 * 0x08);
        buf[i++] = (unsigned char)(0x1F + l_fifth);
        buf[i++] = (unsigned char)(0x00 + l_sixth + l_regbas0 + l_vecval1*8 );
        if ( l_regbas0 == 4 ) buf[i++]=(unsigned char)(0x24);
    } else {
        buf[i++] = (unsigned char)(0x62);
        buf[i++] = (unsigned char)(0xf1 + l_second - l_gp8 * 0x20 - l_ix8 * 0x40 - l_oddgrp1*0x80 - l_2or3grp1 * 0x10);
        buf[i++] = (unsigned char)(0x7c + l_third - l_oddgrp0 * 0x40 - l_vecval0*8);
        buf[i++] = (unsigned char)(0x48 + l_fourth - l_2or3grp0 * 0x08);
        buf[i++] = (unsigned char)(0x1F + l_fifth);
        buf[i++] = (unsigned char)(0x04 + l_sixth + l_vecval1*8 );
        buf[i++] = (unsigned char)(0x00 + l_scaleadj + l_regbas0 + l_regidx*8);
    }

    if ( (l_regbas0 == 5) && (i_displacement==0) )
    {
       /* Registers like rbp/r13 when you have a displacement of 0, we need
          force the single byte of zero to appear. */
        l_forced_offset = 1;
    }
    i += internal_x86_instructions_add_offset( l_place, i, i_displacement, l_forced_offset, l_sizereg, buf );
    if ( i_immediate != LIBXSMM_X86_IMM_UNDEF )
    {
       buf[i++] = (unsigned char)(i_immediate);
    }

    io_generated_code->code_size = i;
    /* *loc = i; */

  } else {
    /* TODO: Debug. This code was just copy/pasted here */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_base[4];
    char l_gp_reg_idx[4];
    char l_instr_name[16];
    char l_broadcast[8];
    char l_masking[16];
    unsigned int l_single_precision = libxsmm_is_x86_vec_instr_single_precision( i_vec_instr );

    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base, 3 );
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name, 15 );

    if (l_single_precision == 0) {
      LIBXSMM_SNPRINTF( l_broadcast, 7, "1to8" );
    } else {
      LIBXSMM_SNPRINTF( l_broadcast, 7, "1to16" );
    }

    if ( i_mask_reg_number != 0 ) {
      /* avoid format-truncation warning due to unsigned int (theoretically) exceeding length of string (l_masking) */
      LIBXSMM_ASSERT_MSG(i_mask_reg_number < 8, "Invalid mask register");
      if ( i_use_zero_masking == 0) {
        if ( io_generated_code->code_type == 0 ) {
          LIBXSMM_SNPRINTF(l_masking, 16, "%%{k%hd%%}", (unsigned short)i_mask_reg_number);
        } else {
          LIBXSMM_SNPRINTF(l_masking, 16, "{k%hd}", (unsigned short)i_mask_reg_number);
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          LIBXSMM_SNPRINTF(l_masking, 16, "%%{k%hd%%}%%{z%%}", (unsigned short)i_mask_reg_number);
        } else {
          LIBXSMM_SNPRINTF(l_masking, 16, "{k%hd}{z}", (unsigned short)i_mask_reg_number);
        }
      }
    }
    else l_masking[0] = (char)0; /* no mask */

    /* build vXYZpd/ps/sd/ss instruction pure register use*/
    if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF ) {
      if ( io_generated_code->code_type == 0 ) {
        if (i_use_broadcast != 0) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s)%%{%s%%}, %%%%%cmm%u, %%%%%cmm%u%s\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, l_masking );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s), %%%%%cmm%u, %%%%%cmm%u%s\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, l_masking );
        }
      } else {
        if (i_use_broadcast != 0) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s) {%s}, %%%cmm%u, %%%cmm%u%s\n", l_instr_name, i_displacement, l_gp_reg_base, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, l_masking );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%%cmm%u, %%%cmm%u%s\n", l_instr_name, i_displacement, l_gp_reg_base, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, l_masking );
        }
      }
    } else {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx, 3 );
      if ( io_generated_code->code_type == 0 ) {
        if (i_use_broadcast != 0) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%%s,%u)%%{%s%%}, %%%%%cmm%u, %%%%%cmm%u%s\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, l_masking );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%%s,%u), %%%%%cmm%u, %%%%%cmm%u%s\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, l_masking );
        }
      } else {
        if (i_use_broadcast != 0) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u) {%s}, %%%cmm%u, %%%cmm%u%s\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, l_broadcast, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, l_masking );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u), %%%cmm%u, %%%cmm%u%s\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1, l_masking );
        }
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_qfma( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_instruction_set,
                                               const unsigned int      i_vec_instr,
                                               const unsigned int      i_gp_reg_base,
                                               const unsigned int      i_gp_reg_idx,
                                               const unsigned int      i_scale,
                                               const int               i_displacement,
                                               const char              i_vector_name,
                                               const unsigned int      i_vec_reg_number_src,
                                               const unsigned int      i_vec_reg_number_dest ) {
  /* @TODO add checks in debug mode */
  if ( i_instruction_set != LIBXSMM_X86_AVX512_KNM ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_NO_AVX512_QFMA );
    return;
  }
  if (libxsmm_is_x86_vec_instr_single_precision( i_vec_instr ) == 0) {
    fprintf( stderr, "LIBXSMM ERROR: QFMA is only supported for single precision\n" );
    exit(-1);
  }
  if (i_vec_reg_number_src%4 != 0) {
    fprintf( stderr, "LIBXSMM ERROR: QFMA source register needs to be a multiple of 4\n" );
    exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /*int i = *loc;*/
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /* unsigned int l_maxsize = 1024; */
    int l_place, l_regc0=0, l_regc1=0, l_regc2=0, l_forced_offset=0;
    int l_sizereg= 1, l_iregnum=0, l_vregnum=0, l_idxnum=0, l_vregdes2=0;
    int l_scalemov = 0;
    int l_instr_off = 0;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }
    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_V4FMADDPS:
          l_instr_off = 0;
          break;
       case LIBXSMM_X86_INSTR_V4FMADDSS:
          l_instr_off = 0x1;
          break;
       case LIBXSMM_X86_INSTR_V4FNMADDPS:
          l_instr_off = 0x10;
          break;
       case LIBXSMM_X86_INSTR_V4FNMADDSS:
          l_instr_off = 0x11;
          break;
       case LIBXSMM_X86_INSTR_VP4DPWSSD:
          l_instr_off = -0x48;
          break;
       case LIBXSMM_X86_INSTR_VP4DPWSSDS:
          l_instr_off = -0x47;
          break;
       default:
          fprintf(stderr, "Strange qmadd instruction\n");
          exit(-1);
    }
    if ( i_gp_reg_base == LIBXSMM_X86_GP_REG_RSP )
    {
       fprintf(stderr, "libxsmm_x86_instruction_vec_compute_qfma isn't designed to work with rsp. Base input off\n");
       exit(-1);
    }
    if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_RSP )
    {
       fprintf(stderr, "libxsmm_x86_instruction_vec_compute_qfma isn't designed to work with rsp. idx input off\n");
       exit(-1);
    }
    if ( /*i_vec_reg_number_dest >= 0 &&*/ i_vec_reg_number_dest <= 7 ) l_regc0 = 0;
    else if ( i_vec_reg_number_dest >= 8 && i_vec_reg_number_dest <= 15 ) l_regc0 = 0x80;
    else if ( i_vec_reg_number_dest >=16 && i_vec_reg_number_dest <= 23 ) l_regc0 = 0x10;
    else if ( i_vec_reg_number_dest >=24 && i_vec_reg_number_dest <= 31 ) l_regc0 = 0x90;
    if ( /*i_vec_reg_number_src >= 0 &&*/ i_vec_reg_number_src <= 7 ) { l_regc1 = 0x40; l_regc2 = 0x08; }
    else if ( i_vec_reg_number_src >= 8 && i_vec_reg_number_src <=15 ) { l_regc1=0; l_regc2 = 0x08; }
    else if ( i_vec_reg_number_src >=16 && i_vec_reg_number_src <=23 ) { l_regc1 =0x40; }
    else if ( i_vec_reg_number_src >=24 && i_vec_reg_number_src <=31 ) { l_regc1 =0; }
    if ( (i_gp_reg_base != LIBXSMM_X86_GP_REG_UNDEF) &&
         (i_gp_reg_base >= LIBXSMM_X86_GP_REG_R8) &&
         (i_gp_reg_base <= LIBXSMM_X86_GP_REG_R15) )
    {
       l_regc0 += 0x20;
    }
    if ( (i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF) &&
         (i_gp_reg_idx >= LIBXSMM_X86_GP_REG_R8) &&
         (i_gp_reg_idx <= LIBXSMM_X86_GP_REG_R15) )
    {
       l_regc0 += 0x40;
    }
    l_iregnum = i_gp_reg_base % 8;
    l_idxnum  = i_gp_reg_idx % 8;
    l_vregnum = (int)(i_vec_reg_number_src/4);
    l_vregnum *= 4;
    l_vregnum = l_vregnum % 8;
    l_vregdes2 = i_vec_reg_number_dest % 8;
    if ( (l_iregnum == 5) && (i_displacement==0) )
    {
       /* Registers like rbp/r13 when you have a displacement of 0, we need */
       /* force the single byte of zero to appear. */
       l_forced_offset=1;
    }
    if ( i_scale == 1 ) l_scalemov = 0x00;
    else if ( i_scale == 2 ) l_scalemov = 0x40;
    else if ( i_scale == 4 ) l_scalemov = 0x80;
    else if ( i_scale == 8 ) l_scalemov = 0xc0;
    else if ( (i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF) &&
         /*(i_gp_reg_idx >= LIBXSMM_X86_GP_REG_RAX) &&*/
         (i_gp_reg_idx <= LIBXSMM_X86_GP_REG_R15) )
    {
       fprintf(stderr, "libxsmm_x86_instruction_vec_compute_qfma has a strange i_scale parameter\n");
       exit(-1);
    }
    buf[i++] = 0x62;
    buf[i++] = (unsigned char)(0xf2 - l_regc0);
    buf[i++] = (unsigned char)(0x3f + l_regc1 - 8*l_vregnum);
    buf[i++] = (unsigned char)(0x40 + l_regc2);
    buf[i++] = (unsigned char)(0x9a + l_instr_off);
    if ( (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF) ||
         /*(i_gp_reg_idx < LIBXSMM_X86_GP_REG_RAX) || */
         (i_gp_reg_idx > LIBXSMM_X86_GP_REG_R15) )
    {
       l_place = i;
       l_sizereg = 16;
       buf[i++] = (unsigned char)(0x00 + l_iregnum + 8*l_vregdes2);
    } else {
       l_place = i;
       buf[i++] = (unsigned char)(0x04 + 8*l_vregdes2);
       l_sizereg = 16;
       buf[i++] = (unsigned char)(l_scalemov + l_iregnum + 8*l_idxnum); /* 0x00 + ... */
    }
/*
    if ( (l_iregnum == LIBXSMM_X86_GP_REG_RSP) || (l_iregnum == LIBXSMM_X86_GP_REG_RBP) )
    {
       buf[i++] = 0x20 + l_iregnum;
    }
*/
    i += internal_x86_instructions_add_offset( l_place, i, i_displacement, l_forced_offset, l_sizereg, buf );

    io_generated_code->code_size = i;
    /* *loc = i; */

  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_base[4];
    char l_gp_reg_idx[4];
    char l_instr_name[16];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base, 3 );
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name, 15 );

    /* build vXYZpd/ps/sd/ss instruction pure register use*/
    if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF ) {
      if ( io_generated_code->code_type == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s), %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, i_vector_name, i_vec_reg_number_src, i_vector_name, i_vec_reg_number_dest );
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, i_vector_name, i_vec_reg_number_src, i_vector_name, i_vec_reg_number_dest );
      }
    } else {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx, 3 );
      if ( io_generated_code->code_type == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%%s,%u), %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_vector_name, i_vec_reg_number_src, i_vector_name, i_vec_reg_number_dest );
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u), %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_vector_name, i_vec_reg_number_src, i_vector_name, i_vec_reg_number_dest );
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
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
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /*int i = *loc;*/
    unsigned int l_maxsize = io_generated_code->buffer_size;
    /*unsigned int l_maxsize = 1024;*/
    int l_vecval0 = i_vec_reg_number_0 % 8;
    int l_vecgrp0 = i_vec_reg_number_0 / 8;
    int l_oddgrp0 = ((l_vecgrp0 % 2)==1);
    int l_vecval1 = i_vec_reg_number_1 % 8;
    int l_vecgrp1 = i_vec_reg_number_1 / 8;
    int l_oddgrp1 = ((l_vecgrp1 % 2)==1);
    int l_vecval2 = i_vec_reg_number_2 % 8;
    int l_vecgrp2 = i_vec_reg_number_2 / 8;
    int l_oddgrp2 = ((l_vecgrp2 % 2)==1);
    int l_extra_byte = 0;
    int l_extra_offset = 0;
    int l_2or3grp0;
    int l_2or3grp1;
    int l_2or3grp2;
    int l_third = 0, l_fifth = 0;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }

    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_VPERM2F128:
          if ( (i_vector_name!='y') && (i_vector_name!='Y') )
          {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: VPERM2F128 only works for ymm\n");
             exit(-1);
          }
          buf[i++] = (unsigned char)(0xc4);
          buf[i++] = (unsigned char)(0xe3 - l_oddgrp0 * 0x20 - l_oddgrp2 * 0x80);
          buf[i++] = (unsigned char)(0x7d - l_oddgrp1 * 0x40 - l_vecval1*8);
          buf[i++] = (unsigned char)(0x06);
          buf[i++] = (unsigned char)(0xc0 + l_vecval0 + l_vecval2*8);
          break;
       case LIBXSMM_X86_INSTR_SHUFPS:
          if ( (i_vector_name!='x') && (i_vector_name!='X') )
          {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: SHUFPS only works for xmm\n");
             exit(-1);
          }
          l_vecgrp0 = 0;
          l_vecgrp1 = 0;
          if ( (i_vec_reg_number_0>=8) && (i_vec_reg_number_0<=15) ) l_vecgrp0 =1;
          if ( (i_vec_reg_number_1>=8) && (i_vec_reg_number_1<=15) ) l_vecgrp1 =1;
          if ( (l_vecgrp0 >= 1) || (l_vecgrp1 >= 1) )     {
             if ( l_vecgrp0 >= 1 ) l_extra_byte += 1;
             if ( l_vecgrp1 >= 1 ) l_extra_byte += 4;
             buf[i++] = (unsigned char)(0x40 + l_extra_byte);
          }
          buf[i++] = (unsigned char)(0x0f);
          buf[i++] = (unsigned char)(0xc6);
          buf[i++] = (unsigned char)(0xc0 + l_vecval0 + l_vecval1*8);
          break;
       case LIBXSMM_X86_INSTR_SHUFPD:
          if ( (i_vector_name!='x') && (i_vector_name!='X') )
          {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: SHUFPD only works for xmm\n");
             exit(-1);
          }
          l_vecgrp0 = 0;
          l_vecgrp1 = 0;
          if ( (i_vec_reg_number_0>=8) && (i_vec_reg_number_0<=15) ) l_vecgrp0 =1;
          if ( (i_vec_reg_number_1>=8) && (i_vec_reg_number_1<=15) ) l_vecgrp1 =1;
          if ( (l_vecgrp0 >= 1) || (l_vecgrp1 >= 1) )     {
             buf[i++] = (unsigned char)(0x66);
             l_extra_byte = 0x22;
             if ( l_vecgrp0 >= 1 ) l_extra_byte += 3;
          }
          buf[i++] = (unsigned char)(0x66 - l_extra_byte);
          buf[i++] = (unsigned char)(0x0f);
          buf[i++] = (unsigned char)(0xc6);
          buf[i++] = (unsigned char)(0xc0 + l_vecval0 + l_vecval1*8);
          break;
       case LIBXSMM_X86_INSTR_VSHUFPS:
          if ( (i_vector_name=='x') || (i_vector_name=='X') )
          {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: VSHUFPS not working for xmm\n");
             exit(-1);
          }
          if ( (i_vector_name=='y') || (i_vector_name=='Y') )
          {
             if ( l_vecgrp0 >= 1 )
             {
                buf[i++] = (unsigned char)(0xc4);
                if ( l_vecgrp2 >= 1 )
                {
                    l_extra_byte = 0x84;
                    l_extra_offset = 0x80;
                } else {
                    l_extra_byte = 0x04;
                }
             }
             buf[i++] = (unsigned char)(0xc5 - l_extra_byte);
             buf[i++] = (unsigned char)(0xfc - l_extra_offset - l_oddgrp0 * 0x80 - l_oddgrp1 * 0x40 - l_oddgrp2 * 0x80 - l_vecval1*8);
             buf[i++] = (unsigned char)(0xc6);
             buf[i++] = (unsigned char)(0xc0 + l_vecval0 + l_vecval2*8);
          } else if ( (i_vector_name=='z') || (i_vector_name=='Z') )
          {
             l_2or3grp0 = (l_vecgrp0>=2);
             l_2or3grp1 = (l_vecgrp1>=2);
             l_2or3grp2 = (l_vecgrp2>=2);
             buf[i++] = (unsigned char)(0x62);
             buf[i++] = (unsigned char)(0xf1 - l_oddgrp0 * 0x20 - l_oddgrp2 * 0x80 - l_2or3grp0 * 0x40 - l_2or3grp2 * 0x10);
             buf[i++] = (unsigned char)(0x7c - l_oddgrp1 * 0x40 - l_vecval1*8);
             buf[i++] = (unsigned char)(0x48 - l_2or3grp1 * 0x08);
             buf[i++] = (unsigned char)(0xc6);
             buf[i++] = (unsigned char)(0xc0 + l_vecval0 + l_vecval2*8);
          } else {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: unknown i_vector_name=%c for VSHUFPS\n",i_vector_name);
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_VSHUFPD:
          if ( (i_vector_name=='x') || (i_vector_name=='X') )
          {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: VSHUFPD not working for xmm\n");
             exit(-1);
          }
          if ( (i_vector_name=='y') || (i_vector_name=='Y') )
          {
             if ( l_vecgrp0 >= 1 )
             {
                buf[i++] = (unsigned char)(0xc4);
                if ( l_vecgrp2 >= 1 )
                {
                    l_extra_byte = 0x84;
                    l_extra_offset = 0x80;
                } else {
                    l_extra_byte = 0x04;
                }
             }
             buf[i++] = (unsigned char)(0xc5 - l_extra_byte);
             /* Only differs from VSHUFS on the 2nd byte here */
             buf[i++] = (unsigned char)(0xfd - l_extra_offset - l_oddgrp0 * 0x80 - l_oddgrp1 * 0x40 - l_oddgrp2 * 0x80 - l_vecval1*8);
             buf[i++] = (unsigned char)(0xc6);
             buf[i++] = (unsigned char)(0xc0 + l_vecval0 + l_vecval2*8);
          } else if ( (i_vector_name=='z') || (i_vector_name=='Z') )
          {
             l_2or3grp0 = (l_vecgrp0>=2);
             l_2or3grp1 = (l_vecgrp1>=2);
             l_2or3grp2 = (l_vecgrp2>=2);
             buf[i++] = (unsigned char)(0x62);
             buf[i++] = (unsigned char)(0xf1 - l_oddgrp0 * 0x20 - l_oddgrp2 * 0x80 - l_2or3grp0 * 0x40 - l_2or3grp2 * 0x10);
             /* Only differs from VSHUFS on the 3rd byte here */
             buf[i++] = (unsigned char)(0xfd - l_oddgrp1 * 0x40 - l_vecval1*8);
             buf[i++] = (unsigned char)(0x48 - l_2or3grp1 * 0x08);
             buf[i++] = (unsigned char)(0xc6);
             buf[i++] = (unsigned char)(0xc0 + l_vecval0 + l_vecval2*8);
          } else {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: unknown i_vector_name=%c for VSHUFPD\n",i_vector_name);
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_VPSRAD:
          if ( i_vec_reg_number_2 != LIBXSMM_X86_VEC_REG_UNDEF ) {
             fprintf(stderr,"libxsmm_x86_instruction_vec_shuffle_reg: shouldn't use vec reg 2 for VPSRAD\n");
             exit(-1);
          }
          l_2or3grp0 = (l_vecgrp0>=2);
          l_2or3grp1 = (l_vecgrp1>=2);
          buf[i++] = (unsigned char)(0x62);
          buf[i++] = (unsigned char)(0xf1 - l_oddgrp0 * 0x20 - l_2or3grp0 * 0x40);
          buf[i++] = (unsigned char)(0x7d - l_oddgrp1 * 0x40 - l_vecval1*8);
          buf[i++] = (unsigned char)(0x48 - l_2or3grp1 * 0x08);
          buf[i++] = (unsigned char)(0x72);
          buf[i++] = (unsigned char)(0xe0 + l_vecval0);
          break;
       case LIBXSMM_X86_INSTR_VPSLLD:
          if ( i_vec_reg_number_2 != LIBXSMM_X86_VEC_REG_UNDEF ) {
             fprintf(stderr,"libxsmm_x86_instruction_vec_shuffle_reg: shouldn't use vec reg 2 for VPSLLD\n");
             exit(-1);
          }
          l_2or3grp0 = (l_vecgrp0>=2);
          l_2or3grp1 = (l_vecgrp1>=2);
          buf[i++] = (unsigned char)(0x62);
          buf[i++] = (unsigned char)(0xf1 - l_oddgrp0 * 0x20 - l_2or3grp0 * 0x40);
          buf[i++] = (unsigned char)(0x7d - l_oddgrp1 * 0x40 - l_vecval1*8);
          buf[i++] = (unsigned char)(0x48 - l_2or3grp1 * 0x08);
          buf[i++] = (unsigned char)(0x72);
          buf[i++] = (unsigned char)(0xf0 + l_vecval0);
          break;
       case LIBXSMM_X86_INSTR_VPSRLD:
          if ( i_vec_reg_number_2 != LIBXSMM_X86_VEC_REG_UNDEF )
          {
             fprintf(stderr,"libxsmm_x86_instruction_vec_shuffle_reg: VPSRLD requires vec2 be undef\n");
             exit(-1);
          }
          if ( (i_vector_name!='z') && (i_vector_name!='Z') )
          {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: VPSRLD only works for zmm\n");
             exit(-1);
          }
          l_2or3grp0 = (l_vecgrp0>=2);
          l_2or3grp1 = (l_vecgrp1>=2);
          buf[i++] = (unsigned char)(0x62);
          buf[i++] = (unsigned char)(0xf1 - l_oddgrp0 * 0x20 - l_2or3grp0 * 0x40);
          buf[i++] = (unsigned char)(0x7d - l_oddgrp1 * 0x40 - l_vecval1*8);
          buf[i++] = (unsigned char)(0x48 - l_2or3grp1 * 0x08);
          buf[i++] = (unsigned char)(0x72);
          buf[i++] = (unsigned char)(0xd0 + l_vecval0);
          break;
       case LIBXSMM_X86_INSTR_VSHUFF64X2:
       case LIBXSMM_X86_INSTR_VSHUFF32X4:
       case LIBXSMM_X86_INSTR_VSHUFI32X4:
       case LIBXSMM_X86_INSTR_VSHUFI64X2:
          l_2or3grp0 = (l_vecgrp0>=2);
          l_2or3grp1 = (l_vecgrp1>=2);
          l_2or3grp2 = (l_vecgrp2>=2);
          if ( (i_vec_instr == LIBXSMM_X86_INSTR_VSHUFF32X4) || (i_vec_instr == LIBXSMM_X86_INSTR_VSHUFI32X4) ) l_third = -0x80;
          if ( (i_vec_instr == LIBXSMM_X86_INSTR_VSHUFI32X4) || (i_vec_instr == LIBXSMM_X86_INSTR_VSHUFI64X2) ) l_fifth = 0x20;

          if ( (i_vector_name!='z') && (i_vector_name!='Z') )
          {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: VSHUF[IF][36][24]X[24] only works for zmm\n");
             exit(-1);
          }
          buf[i++] = (unsigned char)(0x62);
          buf[i++] = (unsigned char)(0xf3 - l_oddgrp0 * 0x20 - l_oddgrp2 * 0x80 - l_2or3grp0 * 0x40 - l_2or3grp2 * 0x10);
          buf[i++] = (unsigned char)(0xfd + l_third - l_oddgrp1 * 0x40 - l_vecval1*8);
          buf[i++] = (unsigned char)(0x48 - l_2or3grp1 * 0x08);
          buf[i++] = (unsigned char)(0x23 + l_fifth);
          buf[i++] = (unsigned char)(0xc0 + l_vecval0 + l_vecval2*8);
          break;
       case LIBXSMM_X86_INSTR_VEXTRACTF32X8:
          l_2or3grp0 = (l_vecgrp0>=2);
          l_2or3grp1 = (l_vecgrp1>=2);
          if ( i_vec_reg_number_2 != LIBXSMM_X86_VEC_REG_UNDEF )
          {
             fprintf(stderr,"libxsmm_x86_instruction_vec_shuffle_reg: VEXTRACTF32X8 requires vec2 be undef\n");
             exit(-1);
          }
          if ( (i_vector_name!='z') && (i_vector_name!='Z') )
          {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: VEXTRACTF32X8 only works for zmm\n");
             exit(-1);
          }
          buf[i++] = (unsigned char)(0x62);
          buf[i++] = (unsigned char)(0xf3 - l_oddgrp0 * 0x80 - l_oddgrp1 * 0x20 - l_2or3grp0 * 0x10 - l_2or3grp1 * 0x40);
          buf[i++] = (unsigned char)(0x7d);
          buf[i++] = (unsigned char)(0x48);
          buf[i++] = (unsigned char)(0x1b);
          buf[i++] = (unsigned char)(0xc0 + l_vecval0*8 + l_vecval1);
          break;
       case LIBXSMM_X86_INSTR_VEXTRACTF64X4:
          l_2or3grp0 = (l_vecgrp0>=2);
          l_2or3grp1 = (l_vecgrp1>=2);
          if ( i_vec_reg_number_2 != LIBXSMM_X86_VEC_REG_UNDEF )
          {
             fprintf(stderr,"libxsmm_x86_instruction_vec_shuffle_reg: VEXTRACTF64x4 requires vec2 be undef\n");
             exit(-1);
          }
          if ( (i_vector_name!='z') && (i_vector_name!='Z') )
          {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg: VEXTRACTF64x4 only works for zmm\n");
             exit(-1);
          }
          buf[i++] = (unsigned char)(0x62);
          buf[i++] = (unsigned char)(0xf3 - l_oddgrp0 * 0x80 - l_oddgrp1 * 0x20 - l_2or3grp0 * 0x10 - l_2or3grp1 * 0x40);
          buf[i++] = (unsigned char)(0xfd);
          buf[i++] = (unsigned char)(0x48);
          buf[i++] = (unsigned char)(0x1b);
          buf[i++] = (unsigned char)(0xc0 + l_vecval0*8 + l_vecval1);
          break;
       default:
          fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_reg doesn't yet do this instruction\n");
          exit(-1);
    }

    /* Every instruction in this group has 1 byte at the end with the operand */
    buf[i++] = (unsigned char)(i_shuffle_operand);

    io_generated_code->code_size = i;
    /* *loc = i; */

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


LIBXSMM_API_INTERN
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
    int l_forced_offset = 0;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
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
          fprintf(stderr, "libxsmm_x86_instruction_vec_move_gathscat: Strange gather/scatter instruction:%u\n",i_vmove_instr);
          exit(-1);
    }
    if ( i_vector_name != 'z' )
    {
       fprintf(stderr, "libxsmm_x86_instruction_vec_move_gathscat: encoder only implemented for zmm registers, but notice that i_vector_name=%c\n",i_vector_name);
       exit(-1);
    }
    if ( i_is_gather == 0 )
    {
       fprintf(stderr, "libxsmm_x86_instruction_vec_move_gathscat: encoder not implemented for scatters yet\n");
       exit(-1);
    }

    { /* open a new scope to avoid warning about mixed declaration and code (C89) */
      int l_regbas0 = i_gp_reg_base % 8;
      int l_gp8     = ((i_gp_reg_base > 7)&&(i_gp_reg_base<=15)?1:0);
      int l_vecval1 = i_vec_reg_number % 8;
      int l_vecgrp1 = i_vec_reg_number / 8;
      int l_oddgrp1 = ((l_vecgrp1 % 2)==1);
      int l_2or3grp1 = (l_vecgrp1>=2);
      int l_vecval0 = i_vec_reg_idx % 8;
      int l_vecgrp0 = i_vec_reg_idx / 8;
      int l_oddgrp0 = ((l_vecgrp0 % 2)==1);
      int l_2or3grp0 = (l_vecgrp0>=2);
      int l_sca=0;

      if (i_scale==2) l_sca=0x40;
      else if (i_scale==4) l_sca=0x80;
      else if (i_scale==8) l_sca=0xc0;

      buf[i++] = (unsigned char)(0x62);
      buf[i++] = (unsigned char)(0xf2 - l_gp8 * 0x20 - l_oddgrp0 * 0x40 - l_oddgrp1 * 0x80 - l_2or3grp1 * 0x10);
      buf[i++] = (unsigned char)(0x7d + l_instr_offset);
      buf[i++] = (unsigned char)(0x48 - l_2or3grp0 * 0x08 + i_mask_reg_number);
      buf[i++] = (unsigned char)(0x92 + l_instr_offset2);
      buf[i++] = (unsigned char)(0x04 + l_vecval1 * 8);
      buf[i++] = (unsigned char)(0x00 + l_sca + l_regbas0 + l_vecval0 * 8);
      if ( (l_regbas0 == 5) && (i_displacement==0) )
      {
          l_forced_offset = 1;
      }
      i += internal_x86_instructions_add_offset( i-2, i, i_displacement, l_forced_offset, l_sizereg, buf );

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
      if ( i_instruction_set >= LIBXSMM_X86_AVX512 ) {
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


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_prefetch( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_prefetch_instr,
                                       const unsigned int      i_gp_reg_base,
                                       const unsigned int      i_gp_reg_idx,
                                       const unsigned int      i_scale,
                                       const int               i_displacement ) {
#if !defined(NDEBUG)
  if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_NO_INDEX_SCALE_ADDR );
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
    int l_instype = 0;
    int l_forced_offset=0;

    int l_regbas0 = i_gp_reg_base % 8;
    int l_gp8 = ((i_gp_reg_base > 7) && (i_gp_reg_base <= 15) ? 1 : 0);
    int l_ix8 = ((i_gp_reg_idx > 7) && (i_gp_reg_idx <= 15) ? 1 : 0);
    int l_sse_preamble = 64;
    int l_place1 = i + 2;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }
    if ( ((int)i_gp_reg_base < LIBXSMM_X86_GP_REG_RAX) ||
         ((int)i_gp_reg_base > LIBXSMM_X86_GP_REG_R15) ||
         (i_gp_reg_base > 15) ||
         ((int)i_gp_reg_base == LIBXSMM_X86_GP_REG_UNDEF) )
    {
       fprintf(stderr, "libxsmm_instruction_prefetch: i_gp_reg_base error in libxsmm_instruction_prefetch\n");
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
          fprintf(stderr, "libxsmm_instruction_prefetch: don't yet do vprefetch0\n");
          exit(-1);
          break;
       case LIBXSMM_X86_INSTR_VPREFETCH1:
          fprintf(stderr, "libxsmm_instruction_prefetch: don't yet do vprefetch1\n");
          exit(-1);
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_prefetch: Strange prefetch instruction: %u\n",i_prefetch_instr);
          exit(-1);
    }

    if ( l_gp8 || l_ix8 )
    {
        if (l_gp8) l_sse_preamble += 1;
#if !defined(NDEBUG) /* TODO: code protected by !defined(NDEBUG) is logically dead */
        LIBXSMM_ASSERT(0 == l_ix8);
        /* coverity[dead_error_line] */
        if (l_ix8) l_sse_preamble += 2;
#endif
        buf[i++] = (unsigned char)l_sse_preamble;
        ++l_place1;
    }

#if !defined(NDEBUG) /* TODO: code protected by !defined(NDEBUG) is logically dead */
    if (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF )
#endif
    {
        LIBXSMM_ASSERT(i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF);
        buf[i++] = 0x0f;
        buf[i++] = 0x18;
        buf[i++] = (unsigned char)(0x10 + l_instype + l_regbas0);
        if ( l_regbas0 == 4 ) buf[i++]=0x24;
    }
#if !defined(NDEBUG)
    else { /* coverity[dead_error_begin] */
        const int l_regidx = i_gp_reg_idx % 8;
        int l_sca = 0;
        if (i_scale == 2) l_sca = 0x40;
        else if (i_scale == 4) l_sca = 0x80;
        else if (i_scale == 8) l_sca = 0xc0;
        buf[i++] = 0x0f;
        buf[i++] = 0x18;
        buf[i++] = (unsigned char)(0x14 + l_instype);
        buf[i++] = (unsigned char)(0x00 + l_sca + l_regbas0 + l_regidx*8);
    }
#else
    LIBXSMM_UNUSED(i_scale);
#endif

    if ( ( l_regbas0 == 5) && (i_displacement==0) )
    {
       /* Registers like rbp/r13 when you have a displacement of 0, we need
 *           force the single byte of zero to appear. */
       l_forced_offset = 1;
    }

    i += internal_x86_instructions_add_offset( l_place1, i, i_displacement, l_forced_offset, 1, buf );

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


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_alu_mem( libxsmm_generated_code* io_generated_code,
                                      const unsigned int     i_alu_instr,
                                      const unsigned int     i_gp_reg_base,
                                      const unsigned int     i_gp_reg_idx,
                                      const unsigned int     i_scale,
                                      const int              i_displacement,
                                      const unsigned int     i_gp_reg_number,
                                      const unsigned int     i_is_store ) {

  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 )
  {
     unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
     int i = io_generated_code->code_size;
     int l_inst = 0x00, l_base = 0x00, l_place2 = i+2;
     int l_regbas0, l_gp8, l_regnum, l_nx8, l_sca = 0, l_forced_offset=0;

     switch ( i_alu_instr ) {
       case LIBXSMM_X86_INSTR_MOVSLQ:
          if ( i_is_store == 1 )
          {
             fprintf(stderr, "libxsmm_instruction_alu_mem: only use LIBXSMM_X86_INSTR_MOVSLQ with loads\n");
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_MOVQ:
          if ( i_is_store == 1 )
          {
             l_inst = 0x26;
          } else {
             l_inst = 0x28;
          }
          break;
       case LIBXSMM_X86_INSTR_LEAQ:
          l_inst = 0x2A;
          break;
       case LIBXSMM_X86_INSTR_MOVL:
          if ( i_is_store == 1 )
          {
             l_inst = 0x26;
          } else {
             l_inst = 0x28;
          }
          l_base = -8;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_alu_mem: Unknown instruction: %u\n", i_alu_instr);
          exit(-1);
     }

     l_regbas0 = i_gp_reg_base % 8;
     l_gp8     = ((i_gp_reg_base > 7)&&(i_gp_reg_base<=15)?1:0);
     l_regnum  = i_gp_reg_number % 8;
     l_nx8     = ((i_gp_reg_number>7)&&(i_gp_reg_number<=15)?1:0);

     if (i_scale==2) l_sca=0x40;
     else if (i_scale==4) l_sca=0x80;
     else if (i_scale==8) l_sca=0xc0;

     if (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF )
     {
         if ((i_alu_instr != LIBXSMM_X86_INSTR_MOVL) || l_gp8 || l_nx8 )
         {
            buf[i++] = (unsigned char)(0x48 + l_base + l_gp8 * 0x01 + l_nx8 * 0x04);
         } else {
            l_place2 = i+1;
         }
         buf[i++] = (unsigned char)(0x63 + l_inst);
         buf[i++] = (unsigned char)(l_sca + l_regbas0 + l_regnum * 0x08);
         if ( l_regbas0 == 4 ) /* rsp or r12 */
         {
            buf[i++] = 0x24;
         }
     } else {
         int l_regidx  = i_gp_reg_idx  % 8;
         int l_ix8     = ((i_gp_reg_idx > 7)&&(i_gp_reg_idx<=15)?1:0);
         if ((i_alu_instr != LIBXSMM_X86_INSTR_MOVL) || l_gp8 || l_nx8 || l_ix8 )
         {
            buf[i++] = (unsigned char)(0x48 + l_base + l_gp8 * 0x01 + l_ix8 * 0x02 + l_nx8 * 0x04);
         } else {
            l_place2 = i+1;
         }
         buf[i++] = (unsigned char)(0x63 + l_inst);
         buf[i++] = (unsigned char)(0x04 + l_regnum * 0x08);
         buf[i++] = (unsigned char)(l_sca + l_regbas0 + l_regidx*8);
     }
     if ( (l_regbas0 == 5) && (i_displacement==0) )
     {
         l_forced_offset = 1;
     }
     i += internal_x86_instructions_add_offset( l_place2, i, i_displacement, l_forced_offset, 1, buf );

     io_generated_code->code_size = i;
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_alu_imm( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_alu_instr,
                                      const unsigned int      i_gp_reg_number,
                                      const long long         i_immediate ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    int l_first = 0;
    int l_second = 0;
    int l_third = 0;
    int l_reg0 = 0;
    int l_extra = 0;
    int l_unsignedadj = 0;
    int l_r8adjment = 1;
    int l_reg0multiplier = 1;

    if (NULL == buf) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
    }

    switch ( i_alu_instr ) {
       case LIBXSMM_X86_INSTR_ADDQ:
          break;
       case LIBXSMM_X86_INSTR_SALQ:
          if ( (i_immediate < 0) || (i_immediate > 127) )
          {
             fprintf(stderr,  "libxsmm_instruction_alu_imm is using an out-of-range immediate for salq.\n"
                              "because other immediates are signed but salq is unsigned. So this code\n"
                              "should be changed if you want an immediate in this range.\n");
             exit(-1);
          }
          l_unsignedadj = 0x3e;
          l_third += 0x20;
          break;
       case LIBXSMM_X86_INSTR_IMUL:
/* Note: we assume that if you call imul in alu_imm you mean: something like imul $3,%rax,%rax. That is, we assume that i_gp_reg_number is used twice */
          l_unsignedadj = -0x18;
          l_extra -= 0x18;
          l_r8adjment = 0x05;
          l_reg0multiplier = 9; /* We are adjusting by 1 and 8 at the same time */
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
          fprintf(stderr, "libxsmm_instruction_alu_imm: Unknown instruction type: %u\n",i_alu_instr);
          exit(-1);
    }
    if ( (i_gp_reg_number > 7) && (i_gp_reg_number <= 15) )
    {
       l_first += l_r8adjment;
       l_reg0 = i_gp_reg_number - 8;
    } else {
       l_reg0 = i_gp_reg_number;
    }
    if ( (i_immediate <= 127) && (i_immediate >= -128) &&
         (i_alu_instr!=LIBXSMM_X86_INSTR_MOVQ) )
    {
       /* one byte (even for 0!) - but never for MOVQ */
       buf[i++] = (unsigned char)(0x48 + l_first);
       buf[i++] = (unsigned char)(0x83 + l_unsignedadj);
       buf[i++] = (unsigned char)(0xc0 + l_third + l_reg0*l_reg0multiplier);
       buf[i++] = (unsigned char)(i_immediate);
    } else {
       /* four bytes */
       unsigned char *l_cptr = (unsigned char *) &i_immediate;
       buf[i++] = (unsigned char)(0x48 + l_first);
       if ( i_gp_reg_number==0 && ((i_alu_instr==LIBXSMM_X86_INSTR_SUBQ) || (i_alu_instr==LIBXSMM_X86_INSTR_CMPQ) || (i_alu_instr==LIBXSMM_X86_INSTR_ADDQ)) )
       {
          /* special case for %rax! */
          buf[i++] = (unsigned char)(0x05 + l_second);
       } else {
          buf[i++] = (unsigned char)(0x81 + l_extra);
          buf[i++] = (unsigned char)(0xc0 + l_third + l_reg0*l_reg0multiplier);
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
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s $%lli, %%%%%s\\n\\t\"\n", l_instr_name, i_immediate, l_gp_reg_name );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%lli, %%%s\n", l_instr_name, i_immediate, l_gp_reg_name );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_alu_imm_i64( libxsmm_generated_code* io_generated_code,
                                          const unsigned int      i_alu_instr,
                                          const unsigned int      i_gp_reg_number,
                                          const size_t            i_immediate ) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    unsigned char *l_cptr = (unsigned char *) &i_immediate;
    int i = io_generated_code->code_size;
    int l_first = 0;
    int l_reg0 = 0;

    if ( i_alu_instr != LIBXSMM_X86_INSTR_MOVQ )
    {
       fprintf(stderr,"How are you doing a 64-byte immediate on instruction: %u\n",i_alu_instr);
       exit(-1);
    }
    if ( /*i_gp_reg_number < 0 ||*/ i_gp_reg_number > 15 )
    {
       fprintf(stderr,"libxsmm_x86_instruction_alu_imm_i64 strange gp reg=%u\n",i_gp_reg_number);
       exit(-1);
    }
    l_reg0 = i_gp_reg_number;
    if ( i_gp_reg_number >= 8 )
    {
       l_first = 1;
       l_reg0 -= 8;
    }
    buf[i++]= (unsigned char)(0x48 + l_first);
    buf[i++]= (unsigned char)(0xb8 + l_reg0);
    buf[i++] = l_cptr[0];
    buf[i++] = l_cptr[1];
    buf[i++] = l_cptr[2];
    buf[i++] = l_cptr[3];
    buf[i++] = l_cptr[4];
    buf[i++] = l_cptr[5];
    buf[i++] = l_cptr[6];
    buf[i++] = l_cptr[7];

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
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s $%" PRIuPTR ", %%%%%s\\n\\t\"\n",
                                       l_instr_name, (uintptr_t)i_immediate, l_gp_reg_name );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%" PRIuPTR ", %%%s\n",
                                       l_instr_name, (uintptr_t)i_immediate, l_gp_reg_name );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_alu_reg( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_alu_instr,
                                      const unsigned int      i_gp_reg_number_src,
                                      const unsigned int      i_gp_reg_number_dest) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    /* int i = *loc; */
    /* unsigned int l_maxsize = io_generated_code->buffer_size;*/
    /* unsigned int l_maxsize = 1024; */

    int l_second = 0;
    int l_third = 0;
    int l_extra_byte = 0;
    int l_reg1 = i_gp_reg_number_src;
    int l_reg0 = i_gp_reg_number_dest;

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
       case LIBXSMM_X86_INSTR_CMOVZ:
          l_second += 0x0e;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVNZ:
          l_second += 0x0e;
          l_third += 0x01;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_POPCNT:
          l_second += 0x0e;
          l_third += 0x74;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_TZCNT:
          l_second += 0x0e;
          l_third += 0x78;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_alu_reg: Not sure what instruction you have in mind: %u\n",i_alu_instr);
          exit(-1);
    }
    {/* open new scope for additional variable declarations (C89) */
      int l_regbas0 = l_reg0 % 8;
      int l_gp8     = ((l_reg0 > 7)&&(l_reg0 <=15)?1:0);
      int l_regnum  = l_reg1 % 8;
      int l_nx8     = ((l_reg1 >7)&&(l_reg1<=15)?1:0);

      if ( (i_alu_instr == LIBXSMM_X86_INSTR_POPCNT) || (i_alu_instr == LIBXSMM_X86_INSTR_TZCNT) ) {
         buf[i++] = (unsigned char)(0xf3);
      }
      buf[i++] = (unsigned char)(0x48 + l_gp8 * 0x01 + l_nx8 * 0x04);
      buf[i++] = (unsigned char)(0x01 + l_second);
      if ( l_extra_byte )
      {
         buf[i++] = (unsigned char)(0x44 + l_third);
      }
      buf[i++] = (unsigned char)(0xc0 + l_regbas0 + 8*l_regnum);

      io_generated_code->code_size = i;
      /* *loc = i; */
    }
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


LIBXSMM_API_INTERN
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
      fprintf(stderr, "libxsmm_instruction_push_reg: push instructions need up to 2 bytes\n");
      exit(-1);
    }
    if ( /*i_gp_reg_number < 0 ||*/ i_gp_reg_number > 15 ) {
      fprintf(stderr, "libxsmm_instruction_push_reg: invalid register\n");
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
    io_generated_code->sf_size += 8;
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
    io_generated_code->sf_size += 8;
  }
}


LIBXSMM_API_INTERN
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
      fprintf(stderr, "libxsmm_instruction_pop_reg: pop instructions need up to 2 bytes\n");
      exit(-1);
    }
    if ( /*i_gp_reg_number < 0 ||*/ i_gp_reg_number > 15 ) {
      fprintf(stderr, "libxsmm_instruction_pop_reg: invalid register\n");
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
    io_generated_code->sf_size -= 8;
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
    io_generated_code->sf_size -= 8;
  }
}


LIBXSMM_API_INTERN
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
    unsigned int l_case = 0;
    int l_regnum0 = i_gp_reg_number % 8;
    int l_nx8 = ((i_gp_reg_number>7)&&(i_gp_reg_number<=15)?1:0);

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }
    switch ( i_mask_instr ) {
       case LIBXSMM_X86_INSTR_KMOVW:
          break;
       case LIBXSMM_X86_INSTR_KMOVB:
          l_case += 1;
          break;
       case LIBXSMM_X86_INSTR_KMOVD:
          l_case += 3;
          break;
       case LIBXSMM_X86_INSTR_KMOVQ:
          l_case += 0x83;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_mask_move: Strange kmov instruction\n");
          exit(-1);
    }
    if ( i_mask_reg_number > 7 )
    {
       fprintf(stderr, "libxsmm_instruction_mask_move: Strange mask number=%u\n",i_mask_reg_number);
       exit(-1);
    }
    if ( l_nx8 || i_mask_instr==LIBXSMM_X86_INSTR_KMOVQ )
    {
       buf[i++] = 0xc4;
       buf[i++] = (unsigned char)(0xe1 - l_nx8*0x20);
       buf[i++] = (unsigned char)(0x78 + l_case);
       buf[i++] = 0x92;
       buf[i++] = (unsigned char)(0xc0 + l_regnum0 + 8*i_mask_reg_number);
    } else {
       buf[i++] = 0xc5;
       buf[i++] = (unsigned char)(0xf8 + l_case);
       buf[i++] = 0x92;
       buf[i++] = (unsigned char)(0xc0 + l_regnum0 + 8*i_mask_reg_number);
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
    if ( (i_gp_reg_number == LIBXSMM_X86_GP_REG_R8  ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R9  ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R10 ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R11 ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R12 ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R13 ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R14 ||
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R15) && (i_mask_instr != LIBXSMM_X86_INSTR_KMOVQ) ) {
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


LIBXSMM_API_INTERN
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
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }
    switch ( i_mask_instr ) {
       case LIBXSMM_X86_INSTR_KXNORW:
          break;
       default:
          fprintf(stderr, "libxsmm_x86_instruction_mask_compute_reg: Strange kmov instruction\n");
          exit(-1);
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


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_register_jump_back_label( libxsmm_generated_code*     io_generated_code,
                                                  libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  /* check if we still have label we can jump to */
  if ( io_loop_label_tracker->label_count == 32 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
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

    io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] = io_loop_label_tracker->label_count+32+1;

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%u:\\n\\t\"\n", io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %u:\n", io_loop_label_tracker->label_address[io_loop_label_tracker->label_count] );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    io_loop_label_tracker->label_count++;
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_jump_back_to_label( libxsmm_generated_code*     io_generated_code,
                                                 const unsigned int          i_jmp_instr,
                                                 libxsmm_loop_label_tracker* io_loop_label_tracker ) {

  /* check that we just handle a valid jump */
  switch ( i_jmp_instr ) {
    case LIBXSMM_X86_INSTR_JL:
    case LIBXSMM_X86_INSTR_JE:
    case LIBXSMM_X86_INSTR_JZ:
    case LIBXSMM_X86_INSTR_JG:
    case LIBXSMM_X86_INSTR_JNE:
    case LIBXSMM_X86_INSTR_JNZ:
    case LIBXSMM_X86_INSTR_JGE:
    case LIBXSMM_X86_INSTR_JLE:
    case LIBXSMM_X86_INSTR_JMP:
      break;
    default:
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUPPORTED_JUMP );
      return;
  }

  /* check if we still have label we can jump to */
  if ( io_loop_label_tracker->label_count == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_NO_JMPLBL_AVAIL );
    return;
  }

  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /*unsigned char *buf = (unsigned char *) io_generated_code->generated_code;*/
    int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int l_lab = --io_loop_label_tracker->label_count;
    int l_val = io_loop_label_tracker->label_address[l_lab];
    /*int l_jmptype, l_dist, l_tmp;*/
    int l_tmp;

    if ( l_maxsize - i < 6 )
    {
       fprintf(stderr, "libxsmm_instruction_jump_back_to_label: Our jump instructions need at most 6 bytes\n");
       exit(-1);
    }

    l_tmp = internal_x86_jumping( io_generated_code, i, l_val, i_jmp_instr );
    io_generated_code->code_size = i + l_tmp;
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


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_register_jump_label( libxsmm_generated_code*     io_generated_code,
                                                  const unsigned int          i_label_no,
                                                  libxsmm_jump_label_tracker* io_jump_label_tracker ) {
  /* check if the label we are trying to set inside of bounds */
  if ( i_label_no >= 32 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  /* check if the label we try to set is still available */
  if ( io_jump_label_tracker->label_address[i_label_no] > 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_JMPLBL_USED );
    return;
  }

  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_ref = 0;
    libxsmm_jump_source l_source = io_jump_label_tracker->label_source[i_label_no];
    /* first added label to tracker */
    io_jump_label_tracker->label_address[i_label_no] = io_generated_code->code_size;
    /* patching all previous references */
    for ( l_ref = 0; l_ref < l_source.ref_count; ++l_ref ) {
      unsigned int l_jmp_instr = l_source.instr_type[l_ref];
      unsigned int l_position =   l_source.instr_addr[l_ref];
#if 0
      int l_tmp =
#endif
      /* This routine just does everything related to jumping. In this case, we know the destination/target */
      internal_x86_jumping ( io_generated_code, l_position, io_generated_code->code_size, l_jmp_instr );
      /* We don't need to forward the bytes here */
    }
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    io_jump_label_tracker->label_address[i_label_no] = i_label_no+1;

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%u:\\n\\t\"\n", io_jump_label_tracker->label_address[i_label_no] );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %u:\n", io_jump_label_tracker->label_address[i_label_no] );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_jump_to_label( libxsmm_generated_code*     io_generated_code,
                                            const unsigned int          i_jmp_instr,
                                            const unsigned int          i_label_no,
                                            libxsmm_jump_label_tracker* io_jump_label_tracker ) {
  unsigned int l_pos;

  /* check if the label we are trying to set inside of bounds */
  if ( (i_label_no < 32) == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  /* check if we still have label we can jump to */
  if ( io_jump_label_tracker->label_source[i_label_no].ref_count == 32-1 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  /* add addr at current position and instruction to tracking structure */
  l_pos = io_jump_label_tracker->label_source[i_label_no].ref_count;
  io_jump_label_tracker->label_source[i_label_no].instr_type[l_pos] = i_jmp_instr;
  io_jump_label_tracker->label_source[i_label_no].instr_addr[l_pos] = io_generated_code->code_size;
  io_jump_label_tracker->label_source[i_label_no].ref_count++;

  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    int l_dest_addr;
    int l_tmp;

    if ( io_jump_label_tracker->label_address[i_label_no] == 0 ) {
      l_dest_addr = -1; /* It's a forward jump to a location we haven't set yet. We'll assume 5-6 bytes */
    } else {
      /* Destination/target address is known here. */
      l_dest_addr = io_jump_label_tracker->label_address[i_label_no];
    }
    l_tmp = internal_x86_jumping ( io_generated_code, io_generated_code->code_size, l_dest_addr, i_jmp_instr );
    io_generated_code->code_size = io_generated_code->code_size + l_tmp; /* l_tmp is the # of bytes needed */

  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[16];
    char l_jmp_dir;
    libxsmm_get_x86_instr_name( i_jmp_instr, l_instr_name, 15 );

    if ( io_jump_label_tracker->label_address[i_label_no] == 0 ) {
      l_jmp_dir = 'f';
    } else {
      l_jmp_dir = 'b';
    }

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %u%c\\n\\t\"\n", l_instr_name, i_label_no+1, l_jmp_dir );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %u%c\n", l_instr_name, i_label_no+1, l_jmp_dir  );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_full_vec_load_of_constants ( libxsmm_generated_code *io_generated_code,
                                                          const unsigned char *i_data,
                                                          const char *i_id,
                                                          const char i_vector_name,
                                                          const unsigned int i_vec_reg_number ) {
  int number_of_bytes_to_load = 0;
  /*int l_regsize_adjustment = 0;*/

  switch ( i_vector_name ) {
    case 'x':
      number_of_bytes_to_load = 16;
      /*l_regsize_adjustment = -4;*/
      break;
    case 'y':
      number_of_bytes_to_load = 32;
      break;
    case 'z':
      number_of_bytes_to_load = 64;
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_full_vec_load_of_constants: strange input for i_vector_name: %c\n",i_vector_name);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 )
  {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    unsigned char *cval = (unsigned char *) &i_data[0];
    int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int j = 0;
    int l_stop = 0;
    int l_regsize_adjustment = 0;
    int l_last_load_location = 0;
    int jmpval = 0;
    int vecval = 0;

    /* @TODO fix max. size error */
    if ( l_maxsize - i < 139 ) {
      fprintf(stderr, "libxsmm_x86_instruction_full_vec_load_of_constants: Most constant jumps need at most 139 bytes\n");
      exit(-1);
    }

#define DISABLE_ALIGNMENT
#ifdef DISABLE_ALIGNMENT
    l_stop = i + 2;
#else
    /* Replace this code with real code to find the right offset "l_stop" so
     * buf[l_stop] has the right alignment, where l_stop >= i+2
     */
    for ( j = i+2, l_stop = -1; (j < i+number_of_bytes_to_load+2) &&
                                (l_stop==-1); j++ )
    {
      if ( ((size_t)&buf[j])%number_of_bytes_to_load == 0 ) { l_stop = j; }
    }
    if ( (l_stop == -1) || (l_stop < i+2) ) {
      fprintf(stderr, "libxsmm_x86_instruction_full_vec_load_of_constants: never found correct alignment\n");
      exit(-1);
    }
    j = l_stop;
#endif

    jmpval = number_of_bytes_to_load + l_stop - (i + 2);
    buf[ i ] = 0xeb;
    buf[i+1] = (unsigned char)jmpval;
    /* Let's insert nops until we reach an aligned address */
    for ( j = i+2; j < l_stop; j++ ) {
      buf[ j ] = 0x90; /* nop */
    }
    i = l_stop;

    for ( j = 0; j < number_of_bytes_to_load; j++ ) {
      buf[ i ] = cval[j];
      i++;
    }
    l_last_load_location = i;
    if ( i_vector_name == 'z' ) {
      buf[ i ] = 0x62;
      if ( i_vec_reg_number <= 7 ) {
        buf[i+1] = 0xf1;
        vecval = i_vec_reg_number;
      } else if ( i_vec_reg_number <= 15 ) {
        buf[i+1] = 0x71;
        vecval = i_vec_reg_number - 8;
      } else if ( i_vec_reg_number <= 23 ) {
        buf[i+1] = 0xe1;
        vecval = i_vec_reg_number - 16;
      } else {
        buf[i+1] = 0x61;
        vecval = i_vec_reg_number - 24;
      }
      buf[i+2] = 0x7c;
      buf[i+3] = 0x48;
      i += 4;
    } else {
      buf[i] = 0xc5;
      if ( i_vec_reg_number <= 7 ) {
        buf[i+1] = (unsigned char)(0xfc + l_regsize_adjustment);
        vecval = i_vec_reg_number;
      } else {
        buf[i+1] = (unsigned char)(0x7c + l_regsize_adjustment);
        vecval = i_vec_reg_number - 8;
      }
      i += 2;
    }

    buf[ i ] = 0x10;
    buf[i+1] = (unsigned char)(0x05 + (8*vecval));
    /* 6 bytes is what we have left to encode in the last_load_location */
    jmpval = -1*(number_of_bytes_to_load + 6 + (i-l_last_load_location) );
    cval = (unsigned char *) &jmpval;
    buf[i+2] = cval[0];
    buf[i+3] = cval[1];
    buf[i+4] = cval[2];
    buf[i+5] = cval[3];
    /* 6 bytes is what we have left to encode in the last_load_location */
    i += 6;

    io_generated_code->code_size = i;
  } else {
    unsigned char *cval = (unsigned char *) &i_data[0];
    int j = 0;
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"jmp .continued_%s\\n\\t\"\n", i_id );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \".data_%s:\\n\\t\"\n", i_id );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      for ( j = 0; j < number_of_bytes_to_load; j += 4 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x\\n\\t\"\n",
                                                                                                        cval[0],cval[1],cval[2],cval[3] );
        cval = cval + 4;
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      }
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \".continued_%s:\\n\\t\"\n", i_id );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"vmovups .data_%s(%%%%rip), %%%%%cmm%u\\n\\t\"\n",
                                                                                                              i_id, i_vector_name, i_vec_reg_number );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       jmp .continued_%s\n", i_id );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       .data_%s:\n", i_id );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      for ( j = 0; j < number_of_bytes_to_load; j += 4 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       .byte 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",
                                                                                                      cval[0],cval[1],cval[2],cval[3] );
        cval = cval + 4;
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      }
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       .continued_%s:\n", i_id );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       vmovups .data_%s(%%rip), %%%cmm%u\n", i_id, i_vector_name, i_vec_reg_number );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_load_arg_to_reg( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_arg_number,
                                              const unsigned int      i_gp_reg_number ) {
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF,
                                   0, io_generated_code->sf_size+8+(8*i_arg_number), i_gp_reg_number, 0 );
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream( libxsmm_generated_code*       io_generated_code,
                                          const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                          unsigned int                  i_prefetch) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    unsigned char* l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (NULL == l_code_buffer || l_max_size < (l_code_size + 9)) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
    }

    /* check for a valid register allocation for input pointers */
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_c ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_C );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a_prefetch ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A_PREF );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b_prefetch ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B_PREF );
      return;
    }

    /* push callee save registers */
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

    /* adjust stack frame size */
    io_generated_code->sf_size += 40;
  } else if ( io_generated_code->code_type == 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_c ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_C );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a_prefetch ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A_PREF );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b_prefetch ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B_PREF );
      return;
    }
    /* push callee save registers */
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

    /* adjust stack frame size */
    io_generated_code->sf_size += 40;

    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       retq\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];

    /* loading a pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "  __asm__ __volatile__(\"movq %%0, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading b pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%1, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading c pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_c, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%2, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading b prefetch pointer in assembly */
    if ( i_prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C ||
         i_prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_name, 3 );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    /* loading a prefetch pointer in assembly */
    } else if ( i_prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_name, 3 );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    /* loading a and b prefetch pointer in assembly */
    } else if ( i_prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C ) {
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


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream( libxsmm_generated_code*       io_generated_code,
                                           const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                           unsigned int                  i_prefetch) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is a very simple System V ABI 64 interface */
    unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (l_max_size < (l_code_size + 10)) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* check for a valid register allocation for input pointers */
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b_prefetch ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B_PREF );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a_prefetch ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A_PREF );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_c ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_C );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A );
      return;
    }

    /* pop callee save registers */
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

    /* adjust stack frame size */
    io_generated_code->sf_size -= 40;

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

    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b_prefetch ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B_PREF );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a_prefetch ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A_PREF );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_c ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_C );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_b ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_B );
      return;
    }
    if ( libxsmm_check_x86_gp_reg_name_callee_save( i_gp_reg_mapping->gp_reg_a ) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_CALLEE_SAVE_A );
      return;
    }

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

    /* adjust stack frame size */
    io_generated_code->sf_size -= 40;

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

    if ( i_prefetch == LIBXSMM_GEMM_PREFETCH_BL2_VIA_C ||
         i_prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD) {
      if ( io_generated_code->arch <= LIBXSMM_X86_AVX2 ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(B_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_b, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(B_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else if ( i_prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ) {
      if ( io_generated_code->arch <= LIBXSMM_X86_AVX2 ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else if ( i_prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C ) {
      if ( io_generated_code->arch <= LIBXSMM_X86_AVX2 ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch), \"m\"(B_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_pre_b, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch), \"m\"(B_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else {
      if ( io_generated_code->arch <= LIBXSMM_X86_AVX2 ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream_transpose( libxsmm_generated_code*                   io_generated_code,
                                                    const unsigned int                        i_gp_reg_a,
                                                    const unsigned int                        i_gp_reg_lda,
                                                    const unsigned int                        i_gp_reg_b,
                                                    const unsigned int                        i_gp_reg_ldb,
                                                    const char*                               i_arch ) {
  LIBXSMM_UNUSED(i_arch);
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    unsigned char* l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (NULL == l_code_buffer || l_max_size < (l_code_size + 9)) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
    }

    /* push rbx */
    l_code_buffer[l_code_size++] = 0x53;
    /* push rbp */
    l_code_buffer[l_code_size++] = 0x55;
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

    /* adjust stack frame size */
    io_generated_code->sf_size += 40;
  } else if ( io_generated_code->code_type == 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%rbx\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%rbp\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r12\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r13\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r14\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%r15\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* adjust stack frame size */
    io_generated_code->sf_size += 40;

    /* @TODO: I don't know if this is the correct placement in the generation process */
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       retq\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];

    /* loading input pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_a, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "  __asm__ __volatile__(\"movq %%0, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading weight pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_lda, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%1, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading output pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_b, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%2, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading input pf pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_ldb, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream_transpose( libxsmm_generated_code*       io_generated_code,
                                                     const char*                   i_arch) {
  /* libxsmm_x86_instruction_close_stream_convolution(io_generated_code, i_arch); */
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is a very simple System V ABI 64 interface */
    unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (NULL == l_code_buffer || l_max_size < (l_code_size + 11)) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
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
    /* pop rbp */
    l_code_buffer[l_code_size++] = 0x5d;
    /* pop rbx */
    l_code_buffer[l_code_size++] = 0x5b;

    /* adjust stack frame size */
    io_generated_code->sf_size -= 40;

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
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%rbp\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* adjust stack frame size */
    io_generated_code->sf_size -= 40;

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


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream_matcopy( libxsmm_generated_code*                   io_generated_code,
                                                  const unsigned int                        i_gp_reg_a,
                                                  const unsigned int                        i_gp_reg_lda,
                                                  const unsigned int                        i_gp_reg_b,
                                                  const unsigned int                        i_gp_reg_ldb,
                                                  const unsigned int                        i_gp_reg_a_pf,
                                                  const unsigned int                        i_gp_reg_b_pf,
                                                  const char*                               i_arch ) {
  LIBXSMM_UNUSED(i_arch);
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    unsigned char* l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (NULL == l_code_buffer || l_max_size < (l_code_size + 9)) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
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

    /* adjust stack frame size */
    io_generated_code->sf_size += 40;
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

    /* adjust stack frame size */
    io_generated_code->sf_size += 40;
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];

    /* loading a pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_a, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "  __asm__ __volatile__(\"movq %%0, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading lda pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_lda, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%1, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading b pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_b, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%2, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading ldb pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_ldb, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading a pf pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_a_pf, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%4, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );

    /* loading b pf pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_b_pf, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       \"movq %%6, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream_matcopy( libxsmm_generated_code*       io_generated_code,
                                                   const char*                   i_arch) {
  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is a very simple System V ABI 64 interface */
    unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (NULL == l_code_buffer || l_max_size < (l_code_size + 10)) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
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

    /* adjust stack frame size */
    io_generated_code->sf_size -= 40;

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

    /* adjust stack frame size */
    io_generated_code->sf_size -= 40;

    /* @TODO: I don't know if this is the correct placement in the generation process */
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       retq\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    char l_new_code[1024];
    int l_max_code_length = 1023;
    int l_code_length = 0;

    if ( (strcmp(i_arch, "wsm") == 0) ||
         (strcmp(i_arch, "snb") == 0) ||
         (strcmp(i_arch, "hsw") == 0) ) {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(aptr), \"m\"(ldaptr), \"m\"(bptr), \"m\"(ldbptr), \"m\"(apfptr), \"m\"(bpfptr) : \"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n");
    } else {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(aptr), \"m\"(ldaptr), \"m\"(bptr), \"m\"(ldbptr), \"m\"(apfptr), \"m\"(bpfptr) : \"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

