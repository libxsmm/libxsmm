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
     fprintf(stderr,"Bogus source location for internal jumping routine: %i\n", i_src_location);
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
     fprintf(stderr,"i_src_location=%i is physically too close to i_dest_location=%i\n",i_src_location,i_dest_location);
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
void libxsmm_x86_instruction_vex_compute_2reg_mem( libxsmm_generated_code*     io_generated_code,
                                                   const unsigned int          i_vec_instr,
                                                   const unsigned int          i_gp_reg_base,
                                                   const unsigned int          i_gp_reg_idx,
                                                   const unsigned int          i_scale,
                                                   const int                   i_displacement,
                                                   const libxsmm_x86_simd_name i_vector_name,
                                                   const unsigned int          i_vec_reg_number_src,
                                                   const unsigned int          i_vec_reg_number_dst )
{
  unsigned int code_head = io_generated_code->code_size;
  unsigned char* code    = (unsigned char *)io_generated_code->generated_code;
  /* easy to address bytes by names */
  unsigned int vexp  = code_head;
  unsigned int p0    = code_head+1;
  unsigned int p1    = code_head+2;
  unsigned int op    = code_head+3;
  unsigned int modrm = code_head+4;
  unsigned int sib   = code_head+5;
  /* the following 3 arrays are look-up table for register names and
     datatype width, we use look up tables to avoid if-statements in
     the encoding path */
  unsigned char tbl_vex_vvvv[16]   = {0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
                                      0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00 };
  unsigned char tbl_scale[9]       = {0x00, 0x00, 0x40, 0x40, 0x80, 0x80, 0x80, 0x80, 0xc0 };
  unsigned char tbl_vl[2]          = {0x00, 0x04};
  /* control variable if we need to encode in SIB mode */
  unsigned char l_have_sib = 0;
  /* index for VL look-ups, zmm is converted to ymm */
  unsigned int l_vl_idx = LIBXSMM_MIN( (unsigned int)i_vector_name, 0x1 );
  /* when having RBP/R13 as base register, we need a SIB byte, even without idx GPR */
  unsigned char l_forced_zdisp8 = 0;
  /* we need a local non-const i_gp_reg_idx copy */
  unsigned int l_gp_reg_idx;
  /* we need a local non-const i_scale copy */
  unsigned int l_scale;

  /* 1st phase: let's compute some static information before starting the
     encoding process */
  /* 1 A) determine if SIB addressing mode is needed */
  if ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_RSP || i_gp_reg_base == LIBXSMM_X86_GP_REG_R12) && (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF) ) {
    l_have_sib = 1;
    l_gp_reg_idx = LIBXSMM_X86_GP_REG_RSP;
    l_scale = 0;
  } else if ( i_gp_reg_idx < 16 ) {
    l_have_sib = 1;
    l_gp_reg_idx = i_gp_reg_idx;
    l_scale = i_scale;
  } else {
    l_have_sib = 0;
    l_gp_reg_idx = 0;
    l_scale = 0;
  }

  /* 1 B) determing if a force zero displacement is needed */
  if ( ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_RBP) || (i_gp_reg_base == LIBXSMM_X86_GP_REG_R13) ) && (i_displacement == 0) ) {
    l_forced_zdisp8 = 1;
  } else {
    l_forced_zdisp8 = 0;
  }

  /* 2nd phase: encoding */
  /* 2 A): writing an insturction template into the byte stream */
  /* @TODO, we right now only encode 3byte VEX */
  /* const VEX prefix */
  code[vexp ] = 0xc4;
  /* p0-op based on instruction value - this is the MMMM field, upper two bits are reseverd to be 00 */
  code[p0   ] = (unsigned char)((i_vec_instr >> 12) & 0x0f);
  /* W-bit and PP prefix */
  code[p1   ] = (unsigned char)((i_vec_instr >> 16) & 0x83);
  /* we are just copying over the OP-code */
  code[op   ] = (unsigned char) i_vec_instr;

  /* 2 B) filling the missing prefix bits based on table look ups */
  /* R */
  code[p0   ] |= (unsigned char)(( i_vec_reg_number_dst < 8 ) ? 0x80 : 0x00);
  /* vvvv and V' */
  code[p1   ] |= (unsigned char)tbl_vex_vvvv[i_vec_reg_number_src];
  /* VL: 128bit,256bit */
  code[p1   ] |= (unsigned char)tbl_vl[l_vl_idx];

  /* 2 C) construction of the Modrm and SIB bytes */
  /* we want to do SIB */
  if ( l_have_sib != 0 ) {
    /* set B */
    code[p0   ] |= (unsigned char)(( i_gp_reg_base < 8 ) ? 0x20 : 0x00);
    /* set X */
    code[p0   ] |= (unsigned char)(( l_gp_reg_idx  < 8 ) ? 0x40 : 0x00);
    /* set registers in modrm and SIB */
    code[modrm] = (unsigned char)(((unsigned char)(i_vec_reg_number_dst << 3)) & 0x38);
    code[modrm] |= (unsigned char)0x04; /* set SIB mode*/
    /* set SIB */
    code[sib  ]  = tbl_scale[l_scale];
    code[sib  ] |= (unsigned char)(((unsigned char)(l_gp_reg_idx << 3)) & 0x38);
    code[sib  ] |= (unsigned char)(((unsigned char) i_gp_reg_base  )    & 0x07);
    /*adjust code head*/
    code_head += 6;
  } else {
    /* B is used and X is unused */
    code[p0   ] |=  (unsigned char)(( i_gp_reg_base < 8 ) ? 0x20 : 0x00);
    /* set registers in modrm */
    code[modrm] = (unsigned char)(((unsigned char)(i_vec_reg_number_dst << 3)) & 0x38);
    code[modrm] |= (unsigned char)(((unsigned char) i_gp_reg_base           )  & 0x07);
    /* adjust coede head*/
    code_head += 5;
  }

  /* 2 D) add displacemnt, if needed */
  if ( (i_displacement != 0) || (l_forced_zdisp8 != 0) ) {
    if ( (i_displacement <= 127) && (i_displacement >=-128) ) {
      code[modrm]       |= (unsigned char)0x40;
      code[code_head++]  = (unsigned char)(i_displacement);
    } else {
      code[modrm]       |= (unsigned char)0x80;
      code[code_head++]  = (unsigned char)(i_displacement);
      code[code_head++]  = (unsigned char)(i_displacement >> 8);
      code[code_head++]  = (unsigned char)(i_displacement >> 16);
      code[code_head++]  = (unsigned char)(i_displacement >> 24);
    }
  } else {}

  /* before return advance to code head ptr in the global structure */
  io_generated_code->code_size = code_head;
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vex_compute_3reg( libxsmm_generated_code*     io_generated_code,
                                               const unsigned int          i_vec_instr,
                                               const libxsmm_x86_simd_name i_vector_name,
                                               const unsigned int          i_vec_reg_number_0,
                                               const unsigned int          i_vec_reg_number_1,
                                               const unsigned int          i_vec_reg_number_2 )
{
  unsigned int code_head = io_generated_code->code_size;
  unsigned char* code    = (unsigned char *)io_generated_code->generated_code;
  /* easy to address bytes by names */
  unsigned int vexp  = code_head;
  unsigned int p0    = code_head+1;
  unsigned int p1    = code_head+2;
  unsigned int op    = code_head+3;
  unsigned int modrm = code_head+4;
  /* the following 8 arrays are look-up table for register names and
     datatype width, we use look up tables to avoid if-statements in
     the encoding path */
  unsigned char tbl_vex_vvvv[16]   = {0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
                                      0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00 };
  unsigned char tbl_vl[2]          = {0x00, 0x04};
  /* index for VL look-ups, zmm is converted to ymm */
  unsigned int l_vl_idx = LIBXSMM_MIN( (unsigned int)i_vector_name, 0x1 );

  /* encoding */
  /* A): writing an insturction template into the byte stream */
  /* const VEX prefix */
  code[vexp ] = 0xc4;
  /* p0-op based on instruction value - this is the MMMM field, upper two bits are reseverd to be 00 */
  code[p0   ] = (unsigned char)((i_vec_instr >> 12) & 0x0f);
  /* W-bit and PP prefix */
  code[p1   ] = (unsigned char)((i_vec_instr >> 16) & 0x83);
  /* we are just copying over the OP-code */
  code[op   ] = (unsigned char) i_vec_instr;

  /* B) filling the missing prefix bits based on table look ups */
  /* R */
  code[p0   ] |= (unsigned char)(( i_vec_reg_number_2 < 8 ) ? 0x80 : 0x00);
  /* B is used and X is unused */
  code[p0   ] |= (unsigned char)(( i_vec_reg_number_0 < 8 ) ? 0x20 : 0x00);
  /* vvvv and V' */
  code[p1   ] |= (unsigned char)tbl_vex_vvvv[i_vec_reg_number_1];
  /* VL: 128bit,256bit */
  code[p1   ] |= (unsigned char)tbl_vl[l_vl_idx];

  /* C) setting modrm, we are in reg-only addressing mode */
  code[modrm]  = (unsigned char)0xc0;
  code[modrm] |= (unsigned char)(((unsigned char)(i_vec_reg_number_2 << 3)) & 0x38);
  code[modrm] |= (unsigned char)(((unsigned char) i_vec_reg_number_0)       & 0x07);

  io_generated_code->code_size = code_head+5;
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_evex_compute_2reg_mem( libxsmm_generated_code*     io_generated_code,
                                                    const unsigned int          i_vec_instr,
                                                    const unsigned int          i_use_broadcast,
                                                    const unsigned int          i_gp_reg_base,
                                                    const unsigned int          i_reg_idx,
                                                    const unsigned int          i_scale,
                                                    const int                   i_displacement,
                                                    const libxsmm_x86_simd_name i_vector_name,
                                                    const unsigned int          i_vec_reg_number_src,
                                                    const unsigned int          i_vec_reg_number_dst,
                                                    const unsigned int          i_mask_reg_number,
                                                    const unsigned int          i_use_zero_masking )
{
  unsigned int code_head = io_generated_code->code_size;
  unsigned char* code    = (unsigned char *)io_generated_code->generated_code;
  /* easy to address bytes by names */
  unsigned int evexp = code_head;
  unsigned int p0    = code_head+1;
  unsigned int p1    = code_head+2;
  unsigned int p2    = code_head+3;
  unsigned int op    = code_head+4;
  unsigned int modrm = code_head+5;
  unsigned int sib   = code_head+6;
  /* the following 8 arrays are look-up table for register names and
     datatype width, we use look up tables to avoid if-statements in
     the encoding path */
  unsigned char tbl_evex_RRp[32]    = {0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
                                      0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
                                      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
  unsigned char tbl_evex_BX[32]     = {0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60,
                                      0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
                                      0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
  unsigned char tbl_evex_vvvv[32]   = {0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
                                      0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00,
                                      0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
                                      0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00 };
  unsigned char tbl_evex_vp[32]     = {0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
                                      0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
  unsigned char tbl_scale[9]        = {0x00, 0x00, 0x40, 0x40, 0x80, 0x80, 0x80, 0x80, 0xc0 };
  unsigned char tbl_vl[3]           = {0x00, 0x20, 0x40};
  unsigned char tbl_disp8div[8]     = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
  unsigned char tbl_disp8divbcst[2] = {0x04, 0x08};
  /* control variable if we need to encode in SIB mode */
  unsigned char l_have_sib = 0;
  /* index for VL look-ups */
  unsigned int l_vl_idx = (unsigned int)i_vector_name;
  /* W-bit */
  unsigned char l_wbit;
  /* displacement 8 divider */
  unsigned char l_disp8div;
  /* when having RBP/R13 as base register, we need a SIB byte, even without idx GPR */
  unsigned char l_forced_zdisp8 = 0;
  /* index into the disp8div table */
  unsigned char l_disp8div_idx;
  /* compressed displacement */
  int l_comp_disp;
  /* we need a local non-const i_reg_idx copy */
  unsigned int l_reg_idx;
  /* we need a local non-const i_scale copy */
  unsigned int l_scale;

  /* 1st phase: let's compute some static information before starting the
     encoding process */
  /* 1 A) handling EVEX compressed displacement */
  if ( i_use_broadcast ) {
    l_wbit     = (unsigned char)((i_vec_instr >> 23) & 1);
    l_disp8div = tbl_disp8divbcst[ l_wbit ];
  } else {
    /* read initial VL=512 calibrated disp8div look up */
    l_disp8div_idx = (unsigned char)((i_vec_instr >> 8) & 0x07);
    /* check we need to adjsut because of VL */
    if ( (unsigned char)((i_vec_instr >> 8) & 0x08) == 8 ) {
      /* Bit 11 is set:  Don't adjust depending on VL */
      l_disp8div = tbl_disp8div[l_disp8div_idx];
    } else {
      /* Bit 11 not set: now we need Spaghetti code */
      if ( i_vector_name != LIBXSMM_X86_SIMD_NAME_ZMM ) {
        if ( (i_vector_name == LIBXSMM_X86_SIMD_NAME_XMM) && (i_vec_instr == 0x20871612) ) {
          /* VMOVDDUP is a special case: eventually FORCE VEX encoding */
          l_disp8div_idx = (unsigned char)(l_disp8div_idx - 3);
        } else {
          /* Changing the index will adjust the powers of 2 automatically */
          if ( l_disp8div_idx < (2 - l_vl_idx) ) {
            l_disp8div_idx = 0;
          } else {
            l_disp8div_idx = (unsigned char)(l_disp8div_idx + l_vl_idx - 2);
          }
        }
      }
      l_disp8div = tbl_disp8div[l_disp8div_idx];
    }
  }

  /* 1 B) determine if SIB addressing mode is needed */
  if ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_RSP || i_gp_reg_base == LIBXSMM_X86_GP_REG_R12) && (i_reg_idx == LIBXSMM_X86_GP_REG_UNDEF) ) {
    l_have_sib = 1;
    l_reg_idx = LIBXSMM_X86_GP_REG_RSP;
    l_scale = 0;
  } else if ( (i_reg_idx < 16) || ( (((i_vec_instr >> 24) & 0x2) == 0x2) && (i_reg_idx < 32) ) ) {
    l_have_sib = 1;
    l_reg_idx = i_reg_idx;
    l_scale = i_scale;
  } else {
    l_have_sib = 0;
    l_reg_idx = 0;
    l_scale = 0;
  }

  /* 1 C) determing if a force zero displacement is needed */
  if ( ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_RBP) || (i_gp_reg_base == LIBXSMM_X86_GP_REG_R13) ) && (i_displacement == 0) ) {
    l_forced_zdisp8 = 1;
  } else {
    l_forced_zdisp8 = 0;
  }

  /* 2nd phase: encoding */
  /* 2 A): writing an insturction template into the byte stream */
  /* const EVEX prefix */
  code[evexp] = 0x62;
  /* p0-op based on instruction value - this is the MMMM field, upper two bits are reseverd to be 00 */
  code[p0   ] = (unsigned char)((i_vec_instr >> 12) & 0x0f);
  /* W-bit and PP prefix */
  code[p1   ] = (unsigned char)((i_vec_instr >> 16) & 0x87);
  /* the fourth prefix byte needs to be compute, let's set it to 0 for now */
  code[p2   ] = 0x00;
  /* we are just copying over the OP-code */
  code[op   ] = (unsigned char) i_vec_instr;

  /* 2 B) filling the missing prefix bits based on table look ups */
  /* R and R' */
  code[p0   ] |= (unsigned char) tbl_evex_RRp[i_vec_reg_number_dst];
  /* vvvv and V' */
  code[p1   ] |= (unsigned char)tbl_evex_vvvv[i_vec_reg_number_src];
  /* incase of gather scatter the V' field is used to extend the idx field for SIB to 32 registers */
  if ( (((i_vec_instr >> 24) & 0x2) == 0x2) ) {
    code[p2   ] |= (unsigned char)  tbl_evex_vp[l_reg_idx];
  } else {
    code[p2   ] |= (unsigned char)  tbl_evex_vp[i_vec_reg_number_src];
  }
  /* VL: 128bit,256bit,512bit */
  code[p2   ] |= (unsigned char)tbl_vl[l_vl_idx];
  /* broadcast */
  code[p2   ] |= (unsigned char)((i_use_broadcast == 0) ? 0x00 : 0x10);
  /* masking */
  code[p2   ] |= (unsigned char)((((i_use_zero_masking != 0) && (i_mask_reg_number != 0)) ? 0x80 : 0x00) | (i_mask_reg_number & 0x07));

  /* 2 C) construction of the Modrm and SIB bytes */
  /* we want to do SIB */
  if ( l_have_sib != 0 ) {
    /* set B */
    code[p0   ] |= (unsigned char)(( i_gp_reg_base < 8 ) ? 0x20 : 0x00);
    /* set X */
    code[p0   ] |= (unsigned char)(( (l_reg_idx & 0x08) == 0x00 ) ? 0x40 : 0x00);
    /* set registers in modrm and SIB */
    code[modrm] = (unsigned char)(((unsigned char)(i_vec_reg_number_dst << 3)) & 0x38);
    code[modrm] |= (unsigned char)0x04; /* set SIB mode*/
    /* set SIB */
    code[sib  ]  = tbl_scale[l_scale];
    code[sib  ] |= (unsigned char)(((unsigned char)(l_reg_idx << 3)) & 0x38);
    code[sib  ] |= (unsigned char)(((unsigned char) i_gp_reg_base  )    & 0x07);
    /*adjust code head*/
    code_head += 7;
  } else {
    /* B and X */
    code[p0   ] |=  (unsigned char)tbl_evex_BX[i_gp_reg_base];
    /* set registers in modrm */
    code[modrm] = (unsigned char)(((unsigned char)(i_vec_reg_number_dst << 3)) & 0x38);
    code[modrm] |= (unsigned char)(((unsigned char) i_gp_reg_base           )  & 0x07);
    /* adjust coede head*/
    code_head += 6;
  }

  /* 2 D) add displacemnt, if needed */
  if ( (i_displacement != 0) || (l_forced_zdisp8 != 0) ) {
    l_comp_disp = i_displacement / l_disp8div;
    if ( (i_displacement % l_disp8div == 0) && (l_comp_disp <= 127) &&
         (l_comp_disp>=-128) ) {
      code[modrm]       |= (unsigned char)0x40;
      code[code_head++]  = (unsigned char)l_comp_disp;
    } else {
      code[modrm]       |= (unsigned char)0x80;
      code[code_head++]  = (unsigned char)(i_displacement);
      code[code_head++]  = (unsigned char)(i_displacement >> 8);
      code[code_head++]  = (unsigned char)(i_displacement >> 16);
      code[code_head++]  = (unsigned char)(i_displacement >> 24);
    }
  } else {}

  /* before return advance to code head ptr in the global structure */
  io_generated_code->code_size = code_head;
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_evex_compute_3reg( libxsmm_generated_code*     io_generated_code,
                                                const unsigned int          i_vec_instr,
                                                const libxsmm_x86_simd_name i_vector_name,
                                                const unsigned int          i_vec_reg_number_0,
                                                const unsigned int          i_vec_reg_number_1,
                                                const unsigned int          i_vec_reg_number_2,
                                                const unsigned int          i_mask_reg_number,
                                                const unsigned int          i_use_zero_masking,
                                                const unsigned char         i_sae_cntl )
{
  unsigned int code_head = io_generated_code->code_size;
  unsigned char* code    = (unsigned char *)io_generated_code->generated_code;
  /* easy to address bytes by names */
  unsigned int evexp = code_head;
  unsigned int p0    = code_head+1;
  unsigned int p1    = code_head+2;
  unsigned int p2    = code_head+3;
  unsigned int op    = code_head+4;
  unsigned int modrm = code_head+5;
  /* the following 8 arrays are look-up table for register names and
     datatype width, we use look up tables to avoid if-statements in
     the encoding path */
  unsigned char tbl_evex_RRp[32]  = {0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
                                     0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
                                     0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
                                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
  unsigned char tbl_evex_BX[32]   = {0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x60,
                                     0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
                                     0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
                                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
  unsigned char tbl_evex_vvvv[32] = {0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
                                     0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00,
                                     0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40,
                                     0x38, 0x30, 0x28, 0x20, 0x18, 0x10, 0x08, 0x00 };
  unsigned char tbl_evex_vp[32]   = {0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
                                     0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
                                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
  unsigned char tbl_vl[3]         = {0x00, 0x20, 0x40};
  /* index for VL look-ups */
  unsigned int l_vl_idx = (unsigned int)i_vector_name;

  /* encoding */
  /* A): writing an insturction template into the byte stream */
  /* const EVEX prefix */
  code[evexp] = 0x62;
  /* p0-op based on instruction value - this is the MMMM field, upper two bits are reseverd to be 00 */
  code[p0   ] = (unsigned char)((i_vec_instr >> 12) & 0x0f);
  /* W-bit and PP prefix */
  code[p1   ] = (unsigned char)((i_vec_instr >> 16) & 0x87);
  /* the fourth prefix byte needs to be compute, let's set it to 0 for now */
  code[p2   ] = 0x00;
  /* we are just copying over the OP-code */
  code[op   ] = (unsigned char) i_vec_instr;

  /* B) filling the missing prefix bits based on table look ups */
  /* R and R' */
  code[p0   ] |= (unsigned char)tbl_evex_RRp[i_vec_reg_number_2];
  /* B and X */
  code[p0   ] |= (unsigned char) tbl_evex_BX[i_vec_reg_number_0];
  /* vvvv and V' */
  code[p1   ] |= (unsigned char)tbl_evex_vvvv[i_vec_reg_number_1];
  code[p2   ] |= (unsigned char)  tbl_evex_vp[i_vec_reg_number_1];
  /* VL: 128bit,256bit,512bit or sae control */
  code[p2   ] |= (unsigned char)((i_sae_cntl == 0) ? tbl_vl[l_vl_idx] : (0x60 & (i_sae_cntl << 4)));
  /* masking */
  code[p2   ] |= (unsigned char)((((i_use_zero_masking != 0) && (i_mask_reg_number != 0)) ? 0x80 : 0x00) | (i_mask_reg_number & 0x07));
  /* enable SAE/RC */
  code[p2   ] |= (unsigned char)((i_sae_cntl == 0) ? 0x00 : 0x10);

  /* C) setting modrm, we are in reg-only addressing mode */
  code[modrm]  = (unsigned char)0xc0;
  code[modrm] |= (unsigned char)(((unsigned char)(i_vec_reg_number_2 << 3)) & 0x38);
  code[modrm] |= (unsigned char)(((unsigned char) i_vec_reg_number_0)       & 0x07);

  io_generated_code->code_size = code_head+6;
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_mask_move( libxsmm_generated_code* io_generated_code,
                                            const unsigned int      i_vmove_instr,
                                            const unsigned int      i_gp_reg_base,
                                            const unsigned int      i_reg_idx,
                                            const unsigned int      i_scale,
                                            const int               i_displacement,
                                            const char              i_vector_name,
                                            const unsigned int      i_vec_reg_number_0,
                                            const unsigned int      i_vec_reg_mask_0,
                                            const unsigned int      i_is_store )
{
  unsigned int l_vmove_instr;

  /* check if passed in a correct instruction */
  switch ( i_vmove_instr ) {
    case LIBXSMM_X86_INSTR_VMASKMOVPD:
    case LIBXSMM_X86_INSTR_VMASKMOVPS:
#if 0
    case LIBXSMM_X86_INSTR_VMASKMOVPD_LD:
    case LIBXSMM_X86_INSTR_VMASKMOVPS_LD:
#endif
    case LIBXSMM_X86_INSTR_VMASKMOVPD_ST:
    case LIBXSMM_X86_INSTR_VMASKMOVPS_ST:
    case LIBXSMM_X86_INSTR_VGATHERDPS_VEX:
    case LIBXSMM_X86_INSTR_VGATHERDPD_VEX:
    case LIBXSMM_X86_INSTR_VGATHERQPS_VEX:
    case LIBXSMM_X86_INSTR_VGATHERQPD_VEX:
    case LIBXSMM_X86_INSTR_VPGATHERDD_VEX:
    case LIBXSMM_X86_INSTR_VPGATHERDQ_VEX:
    case LIBXSMM_X86_INSTR_VPGATHERQD_VEX:
    case LIBXSMM_X86_INSTR_VPGATHERQQ_VEX:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: unexpected instruction number: %u\n", i_vmove_instr);
      exit(-1);
  }

  /* select the code generator REX/VEX/EVEX */
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) &&
       (io_generated_code->code_type > 1) ) {
    libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
    /* check if we have enough code buffer space left */
    if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* as LD/ST semantics have different op codes we need some fix-ups here */
    switch (i_vmove_instr) {
      case LIBXSMM_X86_INSTR_VMASKMOVPD:
        l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMASKMOVPD_LD : LIBXSMM_X86_INSTR_VMASKMOVPD_ST;
        break;
      case LIBXSMM_X86_INSTR_VMASKMOVPS:
        l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMASKMOVPS_LD : LIBXSMM_X86_INSTR_VMASKMOVPS_ST;
        break;
      default:
        l_vmove_instr = i_vmove_instr;
        break;
    }

    /* ceck for gather */
    if ( (((i_vmove_instr >> 24) & 0x2) == 0x2) ) {
      if (i_reg_idx > 15) {
        fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: SIB addressing mode is required for instruction number: %u\n", i_vmove_instr);
        exit(-1);
      }
      if ( (i_vec_reg_mask_0 == i_vec_reg_number_0) || (i_reg_idx == i_vec_reg_number_0) || (i_reg_idx == i_vec_reg_mask_0) ) {
        fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: same register names cannot be used multiple times: %u\n", i_vmove_instr);
        exit(-1);
      }
    }

    /* set simd name */
    switch(i_vector_name) {
      case 'x':
        l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
        break;
      case 'y':
        l_simd_name = LIBXSMM_X86_SIMD_NAME_YMM;
        break;
      default:
        fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: unsupported vlen: %c\n", i_vector_name);
        break;
    }

    /* invoke VEX encoder */
    libxsmm_x86_instruction_vex_compute_2reg_mem ( io_generated_code,
          l_vmove_instr, i_gp_reg_base,
          i_reg_idx, i_scale, i_displacement, l_simd_name,
          i_vec_reg_mask_0, i_vec_reg_number_0 );
  } else if ( io_generated_code->code_type < 2 ) {
    /* add inline/assembly printing */
    fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: ASM/inline ASM is not supported\n");
    exit(-1);
  } else {
    /* general encoder error */
    fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: GENERAL ERROR\n");
    exit(-1);
  }
}
/*
            libxsmm_x86_instruction_vec_move( io_generated_code,
                i_micro_kernel_config->instruction_set,
                LIBXSMM_X86_INSTR_VMOVDQU8,
                i_gp_reg_mapping->gp_reg_c,
                LIBXSMM_X86_GP_REG_UNDEF, 0,
                ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out),
                'z',
                0, 2, 1, 0 );
*/

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_move( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_instruction_set,
                                       const unsigned int      i_vmove_instr,
                                       const unsigned int      i_gp_reg_base,
                                       const unsigned int      i_reg_idx,
                                       const unsigned int      i_scale,
                                       const int               i_displacement,// ((l_n * i_xgemm_desc->ldc) + (l_m * (i_micro_kernel_config->vector_length))) * (i_micro_kernel_config->datatype_size_out)
                                       const char              i_vector_name,//'z'
                                       const unsigned int      i_vec_reg_number_0,//0
                                       const unsigned int      i_mask_reg_number,//2
                                       const unsigned int      i_use_zero_masking,//1
                                       const unsigned int      i_is_store )//0
{
  unsigned int l_vmove_instr;

  /* check if passed in a correct instruction */
  switch ( i_vmove_instr ) {
    case LIBXSMM_X86_INSTR_VMOVAPD:
    case LIBXSMM_X86_INSTR_VMOVAPS:
    case LIBXSMM_X86_INSTR_VMOVUPD:
    case LIBXSMM_X86_INSTR_VMOVUPS:
    case LIBXSMM_X86_INSTR_VMOVSS:
    case LIBXSMM_X86_INSTR_VMOVSD:
    case LIBXSMM_X86_INSTR_VMOVDQA32:
    case LIBXSMM_X86_INSTR_VMOVDQA64:
    case LIBXSMM_X86_INSTR_VMOVDQU8:
    case LIBXSMM_X86_INSTR_VMOVDQU16:
    case LIBXSMM_X86_INSTR_VMOVDQU32:
    case LIBXSMM_X86_INSTR_VMOVDQU64:
#if 0
    case LIBXSMM_X86_INSTR_VMOVAPD_LD:
    case LIBXSMM_X86_INSTR_VMOVAPS_LD:
    case LIBXSMM_X86_INSTR_VMOVUPD_LD:
    case LIBXSMM_X86_INSTR_VMOVUPS_LD:
    case LIBXSMM_X86_INSTR_VMOVSS_LD:
    case LIBXSMM_X86_INSTR_VMOVSD_LD:
    case LIBXSMM_X86_INSTR_VMOVDQA32_LD:
    case LIBXSMM_X86_INSTR_VMOVDQA64_LD:
    case LIBXSMM_X86_INSTR_VMOVDQU8_LD:
    case LIBXSMM_X86_INSTR_VMOVDQU16_LD:
    case LIBXSMM_X86_INSTR_VMOVDQU32_LD:
    case LIBXSMM_X86_INSTR_VMOVDQU64_LD:
#endif
    case LIBXSMM_X86_INSTR_VMOVAPD_ST:
    case LIBXSMM_X86_INSTR_VMOVAPS_ST:
    case LIBXSMM_X86_INSTR_VMOVUPD_ST:
    case LIBXSMM_X86_INSTR_VMOVUPS_ST:
    case LIBXSMM_X86_INSTR_VMOVSS_ST:
    case LIBXSMM_X86_INSTR_VMOVSD_ST:
    case LIBXSMM_X86_INSTR_VMOVDQA32_ST:
    case LIBXSMM_X86_INSTR_VMOVDQA64_ST:
    case LIBXSMM_X86_INSTR_VMOVDQU8_ST:
    case LIBXSMM_X86_INSTR_VMOVDQU16_ST:
    case LIBXSMM_X86_INSTR_VMOVDQU32_ST:
    case LIBXSMM_X86_INSTR_VMOVDQU64_ST:
    case LIBXSMM_X86_INSTR_VPBROADCASTD:
    case LIBXSMM_X86_INSTR_VPBROADCASTQ:
    case LIBXSMM_X86_INSTR_VPBROADCASTQ_VEX:
    case LIBXSMM_X86_INSTR_VPBROADCASTB:
    case LIBXSMM_X86_INSTR_VPBROADCASTW:
    case LIBXSMM_X86_INSTR_VBROADCASTSD:
    case LIBXSMM_X86_INSTR_VBROADCASTSS:
    case LIBXSMM_X86_INSTR_VBROADCASTSD_VEX:
    case LIBXSMM_X86_INSTR_VMOVNTPD:
    case LIBXSMM_X86_INSTR_VMOVNTPS:
    case LIBXSMM_X86_INSTR_VMOVNTDQ:
    case LIBXSMM_X86_INSTR_VPMOVDW:
    case LIBXSMM_X86_INSTR_VPMOVDB:
    case LIBXSMM_X86_INSTR_VPMOVSDB:
    case LIBXSMM_X86_INSTR_VPMOVUSDB:
    case LIBXSMM_X86_INSTR_VPMOVSXWD:
    case LIBXSMM_X86_INSTR_VPMOVZXWD:
    case LIBXSMM_X86_INSTR_VPMOVSXBD:
    case LIBXSMM_X86_INSTR_VPMOVZXBD:
    case LIBXSMM_X86_INSTR_VPMOVUSWB:
    case LIBXSMM_X86_INSTR_VPMOVSWB:
    case LIBXSMM_X86_INSTR_VPMOVWB:
    case LIBXSMM_X86_INSTR_VMOVDDUP:
    case LIBXSMM_X86_INSTR_VBROADCASTI128:
    case LIBXSMM_X86_INSTR_VBROADCASTI32X2:
    case LIBXSMM_X86_INSTR_VBROADCASTI32X4:
    case LIBXSMM_X86_INSTR_VBROADCASTI64X2:
    case LIBXSMM_X86_INSTR_VBROADCASTI32X8:
    case LIBXSMM_X86_INSTR_VBROADCASTI64X4:
    case LIBXSMM_X86_INSTR_VGATHERDPS:
    case LIBXSMM_X86_INSTR_VGATHERDPD:
    case LIBXSMM_X86_INSTR_VGATHERQPS:
    case LIBXSMM_X86_INSTR_VGATHERQPD:
    case LIBXSMM_X86_INSTR_VPGATHERDD:
    case LIBXSMM_X86_INSTR_VPGATHERDQ:
    case LIBXSMM_X86_INSTR_VPGATHERQD:
    case LIBXSMM_X86_INSTR_VPGATHERQQ:
    case LIBXSMM_X86_INSTR_VSCATTERDPS:
    case LIBXSMM_X86_INSTR_VSCATTERDPD:
    case LIBXSMM_X86_INSTR_VSCATTERQPS:
    case LIBXSMM_X86_INSTR_VSCATTERQPD:
    case LIBXSMM_X86_INSTR_VPSCATTERDD:
    case LIBXSMM_X86_INSTR_VPSCATTERDQ:
    case LIBXSMM_X86_INSTR_VPSCATTERQD:
    case LIBXSMM_X86_INSTR_VPSCATTERQQ:
    case LIBXSMM_X86_INSTR_VMOVD_LD:
    case LIBXSMM_X86_INSTR_VMOVQ_LD:
    case LIBXSMM_X86_INSTR_VMOVD_ST:
    case LIBXSMM_X86_INSTR_VMOVQ_ST:
    case LIBXSMM_X86_INSTR_MOVAPD:
    case LIBXSMM_X86_INSTR_MOVUPD:
    case LIBXSMM_X86_INSTR_MOVAPS:
    case LIBXSMM_X86_INSTR_MOVUPS:
    case LIBXSMM_X86_INSTR_MOVSD:
    case LIBXSMM_X86_INSTR_MOVSS:
    case LIBXSMM_X86_INSTR_MOVDDUP:
      break;
    default:
      fprintf(stderr, "libxsmm_instruction_vec_move: unexpected instruction number: %u\n", i_vmove_instr);
      exit(-1);
  }

  /* check for correct streaming stores */
  if ( (i_is_store == 0) && ( (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVNTPD) ||
                              (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVNTPS) ||
                              (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVNTDQ)   )) {
    fprintf(stderr, "libxsmm_instruction_vec_move: streaming stores are only available when setting storing option to true!\n");
    exit(-1);
  }

  /* check that we are not masking 'y' */
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512) && (i_mask_reg_number != 0) ) {
    fprintf(stderr, "libxsmm_instruction_vec_move: Masking is only available for AVX512!\n");
    exit(-1);
  }

  /* check zero masking */
  if ( (i_use_zero_masking != 0) && (i_mask_reg_number != 0) && (i_is_store != 0) ) {
    fprintf(stderr, "libxsmm_instruction_vec_move: zero-masked store cannot operate on memory destination!\n");
    exit(-1);
  }

  /* select the code generator REX/VEX/EVEX */
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) &&
       (io_generated_code->code_type > 1 ) ) {
    /* check if we have enough code buffer space left */
    if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

     /* LD/ST insturction have only 2 operanads */
    if ( ((i_vmove_instr >> 28) & 0x3) == 2 ) {
      unsigned int l_encoder; /* 2=EVEX, 1=VEX, 0=REX */
      unsigned int l_encoder_arch = 2;
      unsigned int l_encoder_instr = ((i_vmove_instr >> 30) & 0x03);

      /* determine encoder */
      //Change by D-
      if ( io_generated_code->arch < LIBXSMM_X86_AVX512) {
        l_encoder_arch = 1;
      } else if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
        l_encoder_arch = 0;
      }
      if ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256) {
        l_encoder_arch = 2;
      }       
      if ( (l_encoder_arch == 2) && ((l_encoder_instr == 3) || (l_encoder_instr == 0)) ) {
        l_encoder = 2;
      } else if ( (l_encoder_arch >= 1) && ((l_encoder_instr == 1) || (l_encoder_instr == 0)) ) {
        l_encoder = 1;
      } else {
        l_encoder = 0;
      }

      /* on Knights platfrom, attempt to fallback to VEX for ymm and xmm VL,
       * will error out in the encoder if instruction doesn't have VEX encoding
       * Core will always take AVX512VL route */
      if ( ( (io_generated_code->arch == LIBXSMM_X86_AVX512_MIC) || (io_generated_code->arch == LIBXSMM_X86_AVX512_KNM) ) &&
           ( (i_vector_name == 'x') || (i_vector_name == 'y') ) && (l_encoder == 2) ) {
        l_encoder = 1;
      }

      /* as LD/ST semantics have different op codes we need some fix-ups here */
      switch (i_vmove_instr) {
        case LIBXSMM_X86_INSTR_VMOVAPD:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVAPD_LD : LIBXSMM_X86_INSTR_VMOVAPD_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVUPD:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVUPD_LD : LIBXSMM_X86_INSTR_VMOVUPD_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVAPS:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVAPS_LD : LIBXSMM_X86_INSTR_VMOVAPS_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVUPS:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVUPS_LD : LIBXSMM_X86_INSTR_VMOVUPS_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVSD:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVSD_LD : LIBXSMM_X86_INSTR_VMOVSD_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVSS:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVSS_LD : LIBXSMM_X86_INSTR_VMOVSS_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVDQA32:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVDQA32_LD : LIBXSMM_X86_INSTR_VMOVDQA32_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVDQA64:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVDQA64_LD : LIBXSMM_X86_INSTR_VMOVDQA64_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVDQU8:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVDQU8_LD : LIBXSMM_X86_INSTR_VMOVDQU8_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVDQU16:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVDQU16_LD : LIBXSMM_X86_INSTR_VMOVDQU16_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVDQU32:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVDQU32_LD : LIBXSMM_X86_INSTR_VMOVDQU32_ST;
          break;
        case LIBXSMM_X86_INSTR_VMOVDQU64:
          l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_VMOVDQU64_LD : LIBXSMM_X86_INSTR_VMOVDQU64_ST;
          break;
        default:
          l_vmove_instr = i_vmove_instr;
          break;
      }

      if ( l_encoder == 2 ) {
        libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
        /* ceck for gather/scatter */
        if ( (((i_vmove_instr >> 24) & 0x2) == 0x2) ) {
          if (i_reg_idx > 32) {
            fprintf(stderr, "libxsmm_instruction_vec_move: SIB addressing mode is required for instruction number: %u\n", i_vmove_instr);
            exit(-1);
          }
          if ( (i_use_zero_masking != 0) || (0 == i_mask_reg_number) ) {
            fprintf(stderr, "libxsmm_instruction_vec_move: merge masking with a valid mask registers (>k0) is required for instrucion number: %u\n", i_vmove_instr);
            exit(-1);
          }
        }

        /* set simd name */
        switch(i_vector_name) {
          case 'x':
            l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
            break;
          case 'y':
            l_simd_name = LIBXSMM_X86_SIMD_NAME_YMM;
            break;
          case 'z':
            l_simd_name = LIBXSMM_X86_SIMD_NAME_ZMM;
            break;
          default:
            fprintf(stderr, "libxsmm_x86_instruction_vec_move: unsupported vlen: %c\n", i_vector_name);
            break;
        }

        libxsmm_x86_instruction_evex_compute_2reg_mem ( io_generated_code,
              l_vmove_instr, 0, i_gp_reg_base,
              i_reg_idx, i_scale, i_displacement, l_simd_name,
              0, i_vec_reg_number_0, i_mask_reg_number, i_use_zero_masking );
      } else if ( l_encoder == 1 ) {
        libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
        /* we need to patch some instructions for VEX from the EVEX header */
        switch (l_vmove_instr) {
          case LIBXSMM_X86_INSTR_VBROADCASTSD:
            l_vmove_instr = LIBXSMM_X86_INSTR_VBROADCASTSD_VEX;
            break;
          default:
            break;
        }

        /* set simd name */
        switch(i_vector_name) {
          case 'x':
            l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
            break;
          case 'y':
            l_simd_name = LIBXSMM_X86_SIMD_NAME_YMM;
            break;
          default:
            fprintf(stderr, "libxsmm_x86_instruction_vec_move: unsupported vlen: %c\n", i_vector_name);
            break;
        }

        libxsmm_x86_instruction_vex_compute_2reg_mem ( io_generated_code,
              l_vmove_instr, i_gp_reg_base,
              i_reg_idx, i_scale, i_displacement, l_simd_name,
              0, i_vec_reg_number_0 );
      } else {
        fprintf(stderr, "libxsmm_x86_instruction_vec_move: No REX encoder available!\n");
        exit(-1);
      }
    } else {
      printf("WARNING: You are calling vec_move with a 3-operand instruction. Are you sure you know what you're doing?\n");
      exit(-1);
    }
    return;
  } else if ( (io_generated_code->arch < LIBXSMM_X86_AVX) &&
              (io_generated_code->code_type > 1 ) ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    int l_aligned=0, l_forced_offset=0;
    int l_num=0, l_sizereg=1;
    int l_scaleadj = 0;
    int l_insert_extra_byte = 0;
    int l_fpadj = 0;
    l_num = i_vec_reg_number_0 / 8;

    switch ( i_vmove_instr ) {
       case LIBXSMM_X86_INSTR_MOVAPD:
          /*l_sse3 = 1;*/
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x18;
          break;
       case LIBXSMM_X86_INSTR_MOVUPD:
          /*l_sse3 = 1;*/
          l_insert_extra_byte = 0x66;
          break;
       case LIBXSMM_X86_INSTR_MOVAPS:
          /*l_sse3 = 1;*/
          l_fpadj = 0x18;
          break;
       case LIBXSMM_X86_INSTR_MOVUPS:
          /*l_sse3 = 1;*/
          break;
       case LIBXSMM_X86_INSTR_MOVSD:
          /*l_sse3 = 1;*/
          l_insert_extra_byte = 0xF2;
          break;
       case LIBXSMM_X86_INSTR_MOVSS:
          /*l_sse3 = 1;*/
          l_insert_extra_byte = 0xF3;
          break;
       case LIBXSMM_X86_INSTR_MOVDDUP:
          /*l_sse3 = 1;*/
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
       default:
          fprintf(stderr, "libxsmm_instruction_vec_move: Exactly what sort of fp regs are you using?\n");
          exit(-1);
    }
    if ( i_is_store == 1 )
    {
       l_aligned += 1;
    }
    {
        /* SSE3 code */
        int l_vecgrp0 = 0;
        int l_vecval0 = i_vec_reg_number_0 % 8;
        int l_place1=i+2;
        int l_regbas0 = i_gp_reg_base % 8;
        int l_regidx =  i_reg_idx % 8;
        int l_gp8 = ((i_gp_reg_base > 7)&&(i_gp_reg_base<=15)?1:0);
        if ( (i_vec_reg_number_0>=8) && (i_vec_reg_number_0<=15) ) l_vecgrp0=1;
        if ( i_is_store ) l_fpadj++;
        if ( l_insert_extra_byte != 0 )
        {
            buf[i++]= (unsigned char)(l_insert_extra_byte);
            ++l_place1;
        }
        if (i_reg_idx == LIBXSMM_X86_GP_REG_UNDEF )
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
          int l_ix8 = ((i_reg_idx > 7) && (i_reg_idx <= 15) ? 1 : 0);
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
void libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( libxsmm_generated_code* io_generated_code,
                                                             const unsigned int      i_vec_instr,
                                                             const char              i_vector_name,
                                                             const unsigned int      i_reg_number_src0,
                                                             const unsigned int      i_reg_number_src1,
                                                             const unsigned int      i_reg_number_dst,
                                                             const unsigned int      i_mask_reg_number,
                                                             const unsigned int      i_mask_cntl,
                                                             const unsigned char     i_sae_cntl,
                                                             const unsigned short    i_imm8 )
{
  /* check if passed in a correct instruction */
  switch ( i_vec_instr ) {
    /* shuffle,extract,blend,unpack,permute */
    case LIBXSMM_X86_INSTR_VSHUFPS:
    case LIBXSMM_X86_INSTR_VSHUFPD:
    case LIBXSMM_X86_INSTR_VPSHUFB:
    case LIBXSMM_X86_INSTR_VPSHUFD:
    case LIBXSMM_X86_INSTR_VPSHUFHW:
    case LIBXSMM_X86_INSTR_VPSHUFLW:
    case LIBXSMM_X86_INSTR_VUNPCKLPD:
    case LIBXSMM_X86_INSTR_VUNPCKLPS:
    case LIBXSMM_X86_INSTR_VUNPCKHPD:
    case LIBXSMM_X86_INSTR_VUNPCKHPS:
    case LIBXSMM_X86_INSTR_VPUNPCKLWD:
    case LIBXSMM_X86_INSTR_VPUNPCKHWD:
    case LIBXSMM_X86_INSTR_VPUNPCKLDQ:
    case LIBXSMM_X86_INSTR_VPUNPCKHDQ:
    case LIBXSMM_X86_INSTR_VPUNPCKLQDQ:
    case LIBXSMM_X86_INSTR_VPUNPCKHQDQ:
    case LIBXSMM_X86_INSTR_VPERMD:
    case LIBXSMM_X86_INSTR_VPERMQ_I:
    case LIBXSMM_X86_INSTR_VPERMPS:
    case LIBXSMM_X86_INSTR_VPERMPD_I:
    case LIBXSMM_X86_INSTR_VPERMILPS:
    case LIBXSMM_X86_INSTR_VPERMILPS_I:
    case LIBXSMM_X86_INSTR_VPERM2F128:
    case LIBXSMM_X86_INSTR_VPERM2I128:
    case LIBXSMM_X86_INSTR_VEXTRACTF128:
    case LIBXSMM_X86_INSTR_VEXTRACTI128:
    case LIBXSMM_X86_INSTR_VPERMILPD_VEX:
    case LIBXSMM_X86_INSTR_VPERMILPD_VEX_I:
    case LIBXSMM_X86_INSTR_VBLENDPD:
    case LIBXSMM_X86_INSTR_VBLENDPS:
    case LIBXSMM_X86_INSTR_VBLENDVPD:
    case LIBXSMM_X86_INSTR_VBLENDVPS:
    case LIBXSMM_X86_INSTR_VPBLENDD:
    case LIBXSMM_X86_INSTR_VPBLENDW:
    case LIBXSMM_X86_INSTR_VPBLENDVB:
    case LIBXSMM_X86_INSTR_VMOVMSKPD:
    case LIBXSMM_X86_INSTR_VMOVMSKPS:
    case LIBXSMM_X86_INSTR_VPMOVMSKB:
    case LIBXSMM_X86_INSTR_VSHUFF32X4:
    case LIBXSMM_X86_INSTR_VSHUFF64X2:
    case LIBXSMM_X86_INSTR_VSHUFI32X4:
    case LIBXSMM_X86_INSTR_VSHUFI64X2:
    case LIBXSMM_X86_INSTR_VEXTRACTF32X4:
    case LIBXSMM_X86_INSTR_VEXTRACTF64X2:
    case LIBXSMM_X86_INSTR_VEXTRACTF32X8:
    case LIBXSMM_X86_INSTR_VEXTRACTF64X4:
    case LIBXSMM_X86_INSTR_VEXTRACTI32X4:
    case LIBXSMM_X86_INSTR_VEXTRACTI64X2:
    case LIBXSMM_X86_INSTR_VEXTRACTI32X8:
    case LIBXSMM_X86_INSTR_VEXTRACTI64X4:
    case LIBXSMM_X86_INSTR_VINSERTI32X4:
    case LIBXSMM_X86_INSTR_VBLENDMPS:
    case LIBXSMM_X86_INSTR_VBLENDMPD:
    case LIBXSMM_X86_INSTR_VPBLENDMB:
    case LIBXSMM_X86_INSTR_VPBLENDMW:
    case LIBXSMM_X86_INSTR_VPBLENDMD:
    case LIBXSMM_X86_INSTR_VPBLENDMQ:
    case LIBXSMM_X86_INSTR_VEXPANDPD:
    case LIBXSMM_X86_INSTR_VEXPANDPS:
    case LIBXSMM_X86_INSTR_VPEXPANDQ:
    case LIBXSMM_X86_INSTR_VPEXPANDD:
    case LIBXSMM_X86_INSTR_VPEXPANDW:
    case LIBXSMM_X86_INSTR_VPEXPANDB:
    case LIBXSMM_X86_INSTR_VPERMW:
    case LIBXSMM_X86_INSTR_VPERMPD:
    case LIBXSMM_X86_INSTR_VPERMT2B:
    case LIBXSMM_X86_INSTR_VPERMT2W:
    case LIBXSMM_X86_INSTR_VPERMT2D:
    case LIBXSMM_X86_INSTR_VPERMT2Q:
    case LIBXSMM_X86_INSTR_VPERMILPD:
    case LIBXSMM_X86_INSTR_VPERMILPD_I:
    case LIBXSMM_X86_INSTR_VFMADD132PS:
    case LIBXSMM_X86_INSTR_VFMADD132PD:
    case LIBXSMM_X86_INSTR_VFMADD213PS:
    case LIBXSMM_X86_INSTR_VFMADD213PD:
    case LIBXSMM_X86_INSTR_VFMADD231PS:
    case LIBXSMM_X86_INSTR_VFMADD231PD:
    case LIBXSMM_X86_INSTR_VFMSUB132PS:
    case LIBXSMM_X86_INSTR_VFMSUB132PD:
    case LIBXSMM_X86_INSTR_VFMSUB213PS:
    case LIBXSMM_X86_INSTR_VFMSUB213PD:
    case LIBXSMM_X86_INSTR_VFMSUB231PS:
    case LIBXSMM_X86_INSTR_VFMSUB231PD:
    case LIBXSMM_X86_INSTR_VFNMADD132PS:
    case LIBXSMM_X86_INSTR_VFNMADD132PD:
    case LIBXSMM_X86_INSTR_VFNMADD213PS:
    case LIBXSMM_X86_INSTR_VFNMADD213PD:
    case LIBXSMM_X86_INSTR_VFNMADD231PS:
    case LIBXSMM_X86_INSTR_VFNMADD231PD:
    case LIBXSMM_X86_INSTR_VFNMSUB132PS:
    case LIBXSMM_X86_INSTR_VFNMSUB132PD:
    case LIBXSMM_X86_INSTR_VFNMSUB213PS:
    case LIBXSMM_X86_INSTR_VFNMSUB213PD:
    case LIBXSMM_X86_INSTR_VFNMSUB231PS:
    case LIBXSMM_X86_INSTR_VFNMSUB231PD:
    case LIBXSMM_X86_INSTR_VFMADD132SD:
    case LIBXSMM_X86_INSTR_VFMADD213SD:
    case LIBXSMM_X86_INSTR_VFMADD231SD:
    case LIBXSMM_X86_INSTR_VFMADD132SS:
    case LIBXSMM_X86_INSTR_VFMADD213SS:
    case LIBXSMM_X86_INSTR_VFMADD231SS:
    case LIBXSMM_X86_INSTR_VFMSUB132SD:
    case LIBXSMM_X86_INSTR_VFMSUB213SD:
    case LIBXSMM_X86_INSTR_VFMSUB231SD:
    case LIBXSMM_X86_INSTR_VFMSUB132SS:
    case LIBXSMM_X86_INSTR_VFMSUB213SS:
    case LIBXSMM_X86_INSTR_VFMSUB231SS:
    case LIBXSMM_X86_INSTR_VFNMADD132SD:
    case LIBXSMM_X86_INSTR_VFNMADD213SD:
    case LIBXSMM_X86_INSTR_VFNMADD231SD:
    case LIBXSMM_X86_INSTR_VFNMADD132SS:
    case LIBXSMM_X86_INSTR_VFNMADD213SS:
    case LIBXSMM_X86_INSTR_VFNMADD231SS:
    case LIBXSMM_X86_INSTR_VFNMSUB132SD:
    case LIBXSMM_X86_INSTR_VFNMSUB213SD:
    case LIBXSMM_X86_INSTR_VFNMSUB231SD:
    case LIBXSMM_X86_INSTR_VFNMSUB132SS:
    case LIBXSMM_X86_INSTR_VFNMSUB213SS:
    case LIBXSMM_X86_INSTR_VFNMSUB231SS:
    case LIBXSMM_X86_INSTR_VROUNDPD:
    case LIBXSMM_X86_INSTR_VROUNDSD:
    case LIBXSMM_X86_INSTR_VROUNDPS:
    case LIBXSMM_X86_INSTR_VROUNDSS:
    case LIBXSMM_X86_INSTR_VRCPPS:
    case LIBXSMM_X86_INSTR_VRCPSS:
    case LIBXSMM_X86_INSTR_VRSQRTPS:
    case LIBXSMM_X86_INSTR_VRSQRTSS:
    case LIBXSMM_X86_INSTR_VRANGEPS:
    case LIBXSMM_X86_INSTR_VRANGEPD:
    case LIBXSMM_X86_INSTR_VRANGESS:
    case LIBXSMM_X86_INSTR_VRANGESD:
    case LIBXSMM_X86_INSTR_VREDUCEPS:
    case LIBXSMM_X86_INSTR_VREDUCEPD:
    case LIBXSMM_X86_INSTR_VREDUCESS:
    case LIBXSMM_X86_INSTR_VREDUCESD:
    case LIBXSMM_X86_INSTR_VRCP14PS:
    case LIBXSMM_X86_INSTR_VRCP14PD:
    case LIBXSMM_X86_INSTR_VRCP14SS:
    case LIBXSMM_X86_INSTR_VRCP14SD:
    case LIBXSMM_X86_INSTR_VRNDSCALEPS:
    case LIBXSMM_X86_INSTR_VRNDSCALEPD:
    case LIBXSMM_X86_INSTR_VRNDSCALESS:
    case LIBXSMM_X86_INSTR_VRNDSCALESD:
    case LIBXSMM_X86_INSTR_VRSQRT14PS:
    case LIBXSMM_X86_INSTR_VRSQRT14PD:
    case LIBXSMM_X86_INSTR_VRSQRT14SS:
    case LIBXSMM_X86_INSTR_VRSQRT14SD:
    case LIBXSMM_X86_INSTR_VSCALEFPS:
    case LIBXSMM_X86_INSTR_VSCALEFPD:
    case LIBXSMM_X86_INSTR_VSCALEFSS:
    case LIBXSMM_X86_INSTR_VSCALEFSD:
    case LIBXSMM_X86_INSTR_VCMPPS:
    case LIBXSMM_X86_INSTR_VCMPSS:
    case LIBXSMM_X86_INSTR_VCMPPD:
    case LIBXSMM_X86_INSTR_VCMPSD:
    case LIBXSMM_X86_INSTR_VPCMPB:
    case LIBXSMM_X86_INSTR_VPCMPUB:
    case LIBXSMM_X86_INSTR_VPCMPW:
    case LIBXSMM_X86_INSTR_VPCMPUW:
    case LIBXSMM_X86_INSTR_VPCMPD:
    case LIBXSMM_X86_INSTR_VPCMPUD:
    case LIBXSMM_X86_INSTR_VPCMPQ:
    case LIBXSMM_X86_INSTR_VPCMPUQ:
    case LIBXSMM_X86_INSTR_VPCMPEQB:
    case LIBXSMM_X86_INSTR_VPCMPEQW:
    case LIBXSMM_X86_INSTR_VPCMPEQD:
    case LIBXSMM_X86_INSTR_VPCMPEQQ:
    case LIBXSMM_X86_INSTR_VPCMPGTB:
    case LIBXSMM_X86_INSTR_VPCMPGTW:
    case LIBXSMM_X86_INSTR_VPCMPGTD:
    case LIBXSMM_X86_INSTR_VPCMPGTQ:
    case LIBXSMM_X86_INSTR_VPCMPESTRI:
    case LIBXSMM_X86_INSTR_VPCMPESTRM:
    case LIBXSMM_X86_INSTR_VPCMPISTRI:
    case LIBXSMM_X86_INSTR_VPCMPISTRM:
    case LIBXSMM_X86_INSTR_VCVTPS2PD:
    case LIBXSMM_X86_INSTR_VCVTPH2PS:
    case LIBXSMM_X86_INSTR_VCVTPS2PH:
    case LIBXSMM_X86_INSTR_VCVTDQ2PS:
    case LIBXSMM_X86_INSTR_VCVTPS2DQ:
    case LIBXSMM_X86_INSTR_VCVTPS2UDQ:
    case LIBXSMM_X86_INSTR_VPMOVDW:
    case LIBXSMM_X86_INSTR_VPMOVSXWD:
    case LIBXSMM_X86_INSTR_VPMOVDB:
    case LIBXSMM_X86_INSTR_VPMOVSDB:
    case LIBXSMM_X86_INSTR_VPMOVUSDB:
    case LIBXSMM_X86_INSTR_VPMOVZXWD:
    case LIBXSMM_X86_INSTR_VPMOVSXBD:
    case LIBXSMM_X86_INSTR_VPMOVZXBD:
    case LIBXSMM_X86_INSTR_VPMOVUSWB:
    case LIBXSMM_X86_INSTR_VPMOVSWB:
    case LIBXSMM_X86_INSTR_VPMOVWB:
    case LIBXSMM_X86_INSTR_VPSLLD_I:
    case LIBXSMM_X86_INSTR_VPSRAD_I:
    case LIBXSMM_X86_INSTR_VPSRLD_I:
    case LIBXSMM_X86_INSTR_VPSLLVW:
    case LIBXSMM_X86_INSTR_VPSLLVD:
    case LIBXSMM_X86_INSTR_VPSLLVQ:
    case LIBXSMM_X86_INSTR_VPSRAVW:
    case LIBXSMM_X86_INSTR_VPSRAVD:
    case LIBXSMM_X86_INSTR_VPSRAVQ:
    case LIBXSMM_X86_INSTR_VPSRLVW:
    case LIBXSMM_X86_INSTR_VPSRLVD:
    case LIBXSMM_X86_INSTR_VPSRLVQ:
    case LIBXSMM_X86_INSTR_VXORPD:
    case LIBXSMM_X86_INSTR_VADDPD:
    case LIBXSMM_X86_INSTR_VMULPD:
    case LIBXSMM_X86_INSTR_VSUBPD:
    case LIBXSMM_X86_INSTR_VDIVPD:
    case LIBXSMM_X86_INSTR_VMINPD:
    case LIBXSMM_X86_INSTR_VMAXPD:
    case LIBXSMM_X86_INSTR_VSQRTPD:
    case LIBXSMM_X86_INSTR_VADDSD:
    case LIBXSMM_X86_INSTR_VMULSD:
    case LIBXSMM_X86_INSTR_VSUBSD:
    case LIBXSMM_X86_INSTR_VDIVSD:
    case LIBXSMM_X86_INSTR_VMINSD:
    case LIBXSMM_X86_INSTR_VMAXSD:
    case LIBXSMM_X86_INSTR_VSQRTSD:
    case LIBXSMM_X86_INSTR_VXORPS:
    case LIBXSMM_X86_INSTR_VADDPS:
    case LIBXSMM_X86_INSTR_VMULPS:
    case LIBXSMM_X86_INSTR_VSUBPS:
    case LIBXSMM_X86_INSTR_VDIVPS:
    case LIBXSMM_X86_INSTR_VMINPS:
    case LIBXSMM_X86_INSTR_VMAXPS:
    case LIBXSMM_X86_INSTR_VSQRTPS:
    case LIBXSMM_X86_INSTR_VMULSS:
    case LIBXSMM_X86_INSTR_VADDSS:
    case LIBXSMM_X86_INSTR_VSUBSS:
    case LIBXSMM_X86_INSTR_VDIVSS:
    case LIBXSMM_X86_INSTR_VMINSS:
    case LIBXSMM_X86_INSTR_VMAXSS:
    case LIBXSMM_X86_INSTR_VSQRTSS:
    case LIBXSMM_X86_INSTR_VPXORD:
    case LIBXSMM_X86_INSTR_VPORD:
    case LIBXSMM_X86_INSTR_VPANDD:
    case LIBXSMM_X86_INSTR_VPANDQ:
    case LIBXSMM_X86_INSTR_VPADDQ:
    case LIBXSMM_X86_INSTR_VPADDB:
    case LIBXSMM_X86_INSTR_VPADDW:
    case LIBXSMM_X86_INSTR_VPADDD:
    case LIBXSMM_X86_INSTR_VPMADDWD:
    case LIBXSMM_X86_INSTR_VPMADDUBSW:
    case LIBXSMM_X86_INSTR_VPADDSW:
    case LIBXSMM_X86_INSTR_VPADDSB:
    case LIBXSMM_X86_INSTR_VPSUBD:
    case LIBXSMM_X86_INSTR_VPMAXSD:
    case LIBXSMM_X86_INSTR_VPMINSD:
    case LIBXSMM_X86_INSTR_VPDPBUSD:
    case LIBXSMM_X86_INSTR_VPDPBUSDS:
    case LIBXSMM_X86_INSTR_VPDPWSSD:
    case LIBXSMM_X86_INSTR_VPDPWSSDS:
    case LIBXSMM_X86_INSTR_VDPBF16PS:
    case LIBXSMM_X86_INSTR_VCVTNEPS2BF16:
    case LIBXSMM_X86_INSTR_VCVTNE2PS2BF16:
    case LIBXSMM_X86_INSTR_VMOVDQU64_LD:
    case LIBXSMM_X86_INSTR_VMOVDQU64_ST:
    case LIBXSMM_X86_INSTR_VMOVDQU32_LD:
    case LIBXSMM_X86_INSTR_VMOVDQU32_ST:
    case LIBXSMM_X86_INSTR_VMOVDQU16_LD:
    case LIBXSMM_X86_INSTR_VMOVDQU16_ST:
    case LIBXSMM_X86_INSTR_VMOVUPS:
    case LIBXSMM_X86_INSTR_VMOVD_LD:
    case LIBXSMM_X86_INSTR_VMOVQ_LD:
    case LIBXSMM_X86_INSTR_VMOVD_ST:
    case LIBXSMM_X86_INSTR_VMOVQ_ST:
    case LIBXSMM_X86_INSTR_VPBROADCASTB_GPR:
    case LIBXSMM_X86_INSTR_VPBROADCASTW_GPR:
    case LIBXSMM_X86_INSTR_VPBROADCASTD_GPR:
    case LIBXSMM_X86_INSTR_VPBROADCASTQ_GPR:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: unexpected instruction number: %u\n", i_vec_instr);
      exit(-1);
  }

  /* check that we are not masking 'y' */
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512) && (i_mask_reg_number != 0) ) {
    fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: Masking is only available for AVX512!\n");
    exit(-1);
  }

  /* check for currently support archs in this encoder */
  if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
    fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: target error!\n");
    exit(-1);
  }

  /* select the code generator REX/VEX/EVEX */
  if ( (i_vec_instr >= 16777216) && (io_generated_code->code_type > 1) ) {
    unsigned int l_encoder; /* 2=EVEX, 1=VEX, 0=REX */
    unsigned int l_encoder_arch = 2;
    unsigned int l_encoder_instr = ((i_vec_instr >> 30) & 0x03);
    unsigned int l_reg_number_src0;
    unsigned int l_reg_number_src1;
    unsigned int l_reg_number_dst;

    /* check if we have enough code buffer space left */
    if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* determine encoder */
    if ( io_generated_code->arch < LIBXSMM_X86_AVX512) {
      l_encoder_arch = 1;
    } else if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
      l_encoder_arch = 0;
    }
    if ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256) {
      l_encoder_arch = 2;
    }
    if ( (l_encoder_arch == 2) && ((l_encoder_instr == 3) || (l_encoder_instr == 0)) ) {
      l_encoder = 2;
    } else if ( (l_encoder_arch >= 1) && ((l_encoder_instr == 1) || (l_encoder_instr == 0)) ) {
      l_encoder = 1;
    } else {
      l_encoder = 0;
    }

    /* check that we have an UNDEF for 2 src operands */
    if ( ((i_vec_instr >> 28) & 3) == 2 ) {
      if ( i_reg_number_src1 != LIBXSMM_X86_VEC_REG_UNDEF ) {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: In case of a 2 src operand instruction (%u), i_reg_number_src1 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        exit(-1);
      }
      l_reg_number_src1 = 0;
    } else {
      l_reg_number_src1 = i_reg_number_src1;
    }

    /* check if we need to flip operands */
    if ( ((i_vec_instr >> 24) & 0x08 ) == 0x08 ) {
      l_reg_number_dst = i_reg_number_src0;
      l_reg_number_src0 = i_reg_number_dst;
    } else {
      l_reg_number_dst = i_reg_number_dst;
      l_reg_number_src0 = i_reg_number_src0;
    }

    /* check if we have op-code extension in modrm/reg */
    if ( ((i_vec_instr >> 24) & 0x04 ) == 0x04 ) {
      if ( ((i_vec_instr >> 28) & 0x3) == 0x2 ) {
        l_reg_number_src1 = i_reg_number_dst;
        l_reg_number_dst = ((i_vec_instr >> 20) & 0x07);
      } else {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: In case of a op-code modrm/reg extended instruciotn (%u), i_reg_number_src1 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        exit(-1);
      }
    }

    /* on Knights platfrom, attempt to fallback to VEX for ymm and xmm VL,
     * will error out in the encoder if instruction doesn't have VEX encoding
     * Core will always take AVX512VL route */
    if ( ( (io_generated_code->arch == LIBXSMM_X86_AVX512_MIC) || (io_generated_code->arch == LIBXSMM_X86_AVX512_KNM) ) &&
         ( (i_vector_name == 'x') || (i_vector_name == 'y') ) && (l_encoder == 2) ) {
      l_encoder = 1;
    }

    /* encode main instruction */
    if ( l_encoder == 2 ) {
      libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;

      /* set simd name */
      switch(i_vector_name) {
        case 'x':
          l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
          break;
        case 'y':
          l_simd_name = LIBXSMM_X86_SIMD_NAME_YMM;
          break;
        case 'z':
          l_simd_name = LIBXSMM_X86_SIMD_NAME_ZMM;
          break;
        default:
          fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: unsupported vlen: %c\n", i_vector_name);
          break;
      }

      libxsmm_x86_instruction_evex_compute_3reg( io_generated_code, i_vec_instr, l_simd_name,
            l_reg_number_src0, l_reg_number_src1, l_reg_number_dst, i_mask_reg_number, i_mask_cntl, i_sae_cntl );
    } else if ( l_encoder == 1 ) {
      libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;

      /* set simd name */
      switch(i_vector_name) {
        case 'x':
          l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
          break;
        case 'y':
          l_simd_name = LIBXSMM_X86_SIMD_NAME_YMM;
          break;
        default:
          fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: unsupported vlen: %c\n", i_vector_name);
          break;
      }

      libxsmm_x86_instruction_vex_compute_3reg( io_generated_code, i_vec_instr, l_simd_name,
            l_reg_number_src0, l_reg_number_src1, l_reg_number_dst );
    } else {
      fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: No REX encoder available!\n");
      exit(-1);
    }

    /* add imm if needed */
    if ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) {
      if ( i_imm8 != LIBXSMM_X86_IMM_UNDEF ) {
        unsigned char* code = (unsigned char *) io_generated_code->generated_code;
        code[io_generated_code->code_size++] = (unsigned char)i_imm8;
      } else {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: imm8 required by instr, but LIBXSMM_X86_IMM_UNDEF was provided!\n");
        exit(-1);
      }
    }
  } else {
    fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: GENERAL ERROR!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_3reg( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_vec_instr,
                                               const char              i_vector_name,
                                               const unsigned int      i_reg_number_src0,
                                               const unsigned int      i_reg_number_src1,
                                               const unsigned int      i_reg_number_dst ) {
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, i_vec_instr, i_vector_name,
                                                           i_reg_number_src0, i_reg_number_src1, i_reg_number_dst,
                                                           0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_3reg_mask( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_instr,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_reg_number_src0,
                                                    const unsigned int      i_reg_number_src1,
                                                    const unsigned int      i_reg_number_dst,
                                                    const unsigned int      i_mask_reg_number,
                                                    const unsigned int      i_mask_cntl ) {
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, i_vec_instr, i_vector_name,
                                                           i_reg_number_src0, i_reg_number_src1, i_reg_number_dst,
                                                           i_mask_reg_number, i_mask_cntl, 0, LIBXSMM_X86_IMM_UNDEF );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_3reg_imm8( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_instr,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_reg_number_src0,
                                                    const unsigned int      i_reg_number_src1,
                                                    const unsigned int      i_reg_number_dst,
                                                    const unsigned short    i_imm8 ) {
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, i_vec_instr, i_vector_name,
                                                           i_reg_number_src0, i_reg_number_src1, i_reg_number_dst,
                                                           0, 0, 0, i_imm8 );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( libxsmm_generated_code* io_generated_code,
                                                             const unsigned int      i_vec_instr,
                                                             const char              i_vector_name,
                                                             const unsigned int      i_reg_number_src0,
                                                             const unsigned int      i_reg_number_dst,
                                                             const unsigned int      i_mask_reg_number,
                                                             const unsigned int      i_mask_cntl,
                                                             const unsigned char     i_sae_cntl,
                                                             const unsigned short    i_imm8 ) {
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, i_vec_instr, i_vector_name,
                                                           i_reg_number_src0, LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
                                                           i_mask_reg_number, i_mask_cntl, i_sae_cntl, i_imm8 );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_2reg( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_vec_instr,
                                               const char              i_vector_name,
                                               const unsigned int      i_reg_number_src0,
                                               const unsigned int      i_reg_number_dst ) {
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, i_vec_instr, i_vector_name,
                                                           i_reg_number_src0, LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
                                                           0, 0, 0, LIBXSMM_X86_IMM_UNDEF );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_2reg_mask( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_instr,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_reg_number_src0,
                                                    const unsigned int      i_reg_number_dst,
                                                    const unsigned int      i_mask_reg_number,
                                                    const unsigned int      i_mask_cntl) {
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, i_vec_instr, i_vector_name,
                                                           i_reg_number_src0, LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
                                                           i_mask_reg_number, i_mask_cntl, 0, LIBXSMM_X86_IMM_UNDEF );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_2reg_imm8( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_instr,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_reg_number_src0,
                                                    const unsigned int      i_reg_number_dst,
                                                    const unsigned short    i_imm8 ) {
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, i_vec_instr, i_vector_name,
                                                           i_reg_number_src0, LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
                                                           0, 0, 0, i_imm8 );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( libxsmm_generated_code* io_generated_code,
                                                             const unsigned int      i_vec_instr,
                                                             const char              i_vector_name,
                                                             const unsigned int      i_gp_reg_base,
                                                             const unsigned int      i_gp_reg_idx,
                                                             const unsigned int      i_scale,
                                                             const int               i_displacement,
                                                             const unsigned int      i_use_broadcast,
                                                             const unsigned int      i_reg_number_src1,
                                                             const unsigned int      i_reg_number_dst,
                                                             const unsigned int      i_mask_reg_number,
                                                             const unsigned int      i_mask_rnd_exp_cntl,
                                                             const unsigned short    i_imm8 )
{
  /* check if passed in a correct instruction */
  switch ( i_vec_instr ) {
    /* shuffle,extract,blend,unpack,permute */
    case LIBXSMM_X86_INSTR_VSHUFPS:
    case LIBXSMM_X86_INSTR_VSHUFPD:
    case LIBXSMM_X86_INSTR_VPSHUFB:
    case LIBXSMM_X86_INSTR_VPSHUFD:
    case LIBXSMM_X86_INSTR_VPSHUFHW:
    case LIBXSMM_X86_INSTR_VPSHUFLW:
    case LIBXSMM_X86_INSTR_VSHUFF32X4:
    case LIBXSMM_X86_INSTR_VSHUFF64X2:
    case LIBXSMM_X86_INSTR_VSHUFI32X4:
    case LIBXSMM_X86_INSTR_VSHUFI64X2:
    case LIBXSMM_X86_INSTR_VEXTRACTF128:
    case LIBXSMM_X86_INSTR_VEXTRACTI128:
    case LIBXSMM_X86_INSTR_VEXTRACTF32X4:
    case LIBXSMM_X86_INSTR_VEXTRACTF64X2:
    case LIBXSMM_X86_INSTR_VEXTRACTF32X8:
    case LIBXSMM_X86_INSTR_VEXTRACTF64X4:
    case LIBXSMM_X86_INSTR_VEXTRACTI32X4:
    case LIBXSMM_X86_INSTR_VEXTRACTI64X2:
    case LIBXSMM_X86_INSTR_VEXTRACTI32X8:
    case LIBXSMM_X86_INSTR_VEXTRACTI64X4:
    case LIBXSMM_X86_INSTR_VINSERTI32X4:
    case LIBXSMM_X86_INSTR_VBLENDMPS:
    case LIBXSMM_X86_INSTR_VBLENDMPD:
    case LIBXSMM_X86_INSTR_VPBLENDMB:
    case LIBXSMM_X86_INSTR_VPBLENDMW:
    case LIBXSMM_X86_INSTR_VPBLENDMD:
    case LIBXSMM_X86_INSTR_VPBLENDMQ:
    case LIBXSMM_X86_INSTR_VEXPANDPD:
    case LIBXSMM_X86_INSTR_VEXPANDPS:
    case LIBXSMM_X86_INSTR_VPEXPANDQ:
    case LIBXSMM_X86_INSTR_VPEXPANDD:
    case LIBXSMM_X86_INSTR_VPEXPANDW:
    case LIBXSMM_X86_INSTR_VPEXPANDB:
    case LIBXSMM_X86_INSTR_VUNPCKLPD:
    case LIBXSMM_X86_INSTR_VUNPCKLPS:
    case LIBXSMM_X86_INSTR_VUNPCKHPD:
    case LIBXSMM_X86_INSTR_VUNPCKHPS:
    case LIBXSMM_X86_INSTR_VPUNPCKLWD:
    case LIBXSMM_X86_INSTR_VPUNPCKHWD:
    case LIBXSMM_X86_INSTR_VPUNPCKLDQ:
    case LIBXSMM_X86_INSTR_VPUNPCKHDQ:
    case LIBXSMM_X86_INSTR_VPUNPCKLQDQ:
    case LIBXSMM_X86_INSTR_VPUNPCKHQDQ:
    case LIBXSMM_X86_INSTR_VPERM2F128:
    case LIBXSMM_X86_INSTR_VPERM2I128:
    case LIBXSMM_X86_INSTR_VPERMW:
    case LIBXSMM_X86_INSTR_VPERMD:
    case LIBXSMM_X86_INSTR_VPERMQ_I:
    case LIBXSMM_X86_INSTR_VPERMT2B:
    case LIBXSMM_X86_INSTR_VPERMT2W:
    case LIBXSMM_X86_INSTR_VPERMT2D:
    case LIBXSMM_X86_INSTR_VPERMT2Q:
    case LIBXSMM_X86_INSTR_VFMADD132PS:
    case LIBXSMM_X86_INSTR_VFMADD132PD:
    case LIBXSMM_X86_INSTR_VFMADD213PS:
    case LIBXSMM_X86_INSTR_VFMADD213PD:
    case LIBXSMM_X86_INSTR_VFMADD231PS:
    case LIBXSMM_X86_INSTR_VFMADD231PD:
    case LIBXSMM_X86_INSTR_VFMSUB132PS:
    case LIBXSMM_X86_INSTR_VFMSUB132PD:
    case LIBXSMM_X86_INSTR_VFMSUB213PS:
    case LIBXSMM_X86_INSTR_VFMSUB213PD:
    case LIBXSMM_X86_INSTR_VFMSUB231PS:
    case LIBXSMM_X86_INSTR_VFMSUB231PD:
    case LIBXSMM_X86_INSTR_VFNMADD132PS:
    case LIBXSMM_X86_INSTR_VFNMADD132PD:
    case LIBXSMM_X86_INSTR_VFNMADD213PS:
    case LIBXSMM_X86_INSTR_VFNMADD213PD:
    case LIBXSMM_X86_INSTR_VFNMADD231PS:
    case LIBXSMM_X86_INSTR_VFNMADD231PD:
    case LIBXSMM_X86_INSTR_VFNMSUB132PS:
    case LIBXSMM_X86_INSTR_VFNMSUB132PD:
    case LIBXSMM_X86_INSTR_VFNMSUB213PS:
    case LIBXSMM_X86_INSTR_VFNMSUB213PD:
    case LIBXSMM_X86_INSTR_VFNMSUB231PS:
    case LIBXSMM_X86_INSTR_VFNMSUB231PD:
    case LIBXSMM_X86_INSTR_VFMADD132SD:
    case LIBXSMM_X86_INSTR_VFMADD213SD:
    case LIBXSMM_X86_INSTR_VFMADD231SD:
    case LIBXSMM_X86_INSTR_VFMADD132SS:
    case LIBXSMM_X86_INSTR_VFMADD213SS:
    case LIBXSMM_X86_INSTR_VFMADD231SS:
    case LIBXSMM_X86_INSTR_VFMSUB132SD:
    case LIBXSMM_X86_INSTR_VFMSUB213SD:
    case LIBXSMM_X86_INSTR_VFMSUB231SD:
    case LIBXSMM_X86_INSTR_VFMSUB132SS:
    case LIBXSMM_X86_INSTR_VFMSUB213SS:
    case LIBXSMM_X86_INSTR_VFMSUB231SS:
    case LIBXSMM_X86_INSTR_VFNMADD132SD:
    case LIBXSMM_X86_INSTR_VFNMADD213SD:
    case LIBXSMM_X86_INSTR_VFNMADD231SD:
    case LIBXSMM_X86_INSTR_VFNMADD132SS:
    case LIBXSMM_X86_INSTR_VFNMADD213SS:
    case LIBXSMM_X86_INSTR_VFNMADD231SS:
    case LIBXSMM_X86_INSTR_VFNMSUB132SD:
    case LIBXSMM_X86_INSTR_VFNMSUB213SD:
    case LIBXSMM_X86_INSTR_VFNMSUB231SD:
    case LIBXSMM_X86_INSTR_VFNMSUB132SS:
    case LIBXSMM_X86_INSTR_VFNMSUB213SS:
    case LIBXSMM_X86_INSTR_VFNMSUB231SS:
    case LIBXSMM_X86_INSTR_VRANGEPS:
    case LIBXSMM_X86_INSTR_VRANGEPD:
    case LIBXSMM_X86_INSTR_VRANGESS:
    case LIBXSMM_X86_INSTR_VRANGESD:
    case LIBXSMM_X86_INSTR_VREDUCEPS:
    case LIBXSMM_X86_INSTR_VREDUCEPD:
    case LIBXSMM_X86_INSTR_VREDUCESS:
    case LIBXSMM_X86_INSTR_VREDUCESD:
    case LIBXSMM_X86_INSTR_VRCP14PS:
    case LIBXSMM_X86_INSTR_VRCP14PD:
    case LIBXSMM_X86_INSTR_VRCP14SS:
    case LIBXSMM_X86_INSTR_VRCP14SD:
    case LIBXSMM_X86_INSTR_VRNDSCALEPS:
    case LIBXSMM_X86_INSTR_VRNDSCALEPD:
    case LIBXSMM_X86_INSTR_VRNDSCALESS:
    case LIBXSMM_X86_INSTR_VRNDSCALESD:
    case LIBXSMM_X86_INSTR_VRSQRT14PS:
    case LIBXSMM_X86_INSTR_VRSQRT14PD:
    case LIBXSMM_X86_INSTR_VRSQRT14SS:
    case LIBXSMM_X86_INSTR_VRSQRT14SD:
    case LIBXSMM_X86_INSTR_VSCALEFPS:
    case LIBXSMM_X86_INSTR_VSCALEFPD:
    case LIBXSMM_X86_INSTR_VSCALEFSS:
    case LIBXSMM_X86_INSTR_VSCALEFSD:
    case LIBXSMM_X86_INSTR_VCMPPS:
    case LIBXSMM_X86_INSTR_VCMPSS:
    case LIBXSMM_X86_INSTR_VCMPPD:
    case LIBXSMM_X86_INSTR_VCMPSD:
    case LIBXSMM_X86_INSTR_VPCMPB:
    case LIBXSMM_X86_INSTR_VPCMPUB:
    case LIBXSMM_X86_INSTR_VPCMPW:
    case LIBXSMM_X86_INSTR_VPCMPUW:
    case LIBXSMM_X86_INSTR_VPCMPD:
    case LIBXSMM_X86_INSTR_VPCMPUD:
    case LIBXSMM_X86_INSTR_VPCMPQ:
    case LIBXSMM_X86_INSTR_VPCMPUQ:
    case LIBXSMM_X86_INSTR_VPCMPEQB:
    case LIBXSMM_X86_INSTR_VPCMPEQW:
    case LIBXSMM_X86_INSTR_VPCMPEQD:
    case LIBXSMM_X86_INSTR_VPCMPEQQ:
    case LIBXSMM_X86_INSTR_VPCMPGTB:
    case LIBXSMM_X86_INSTR_VPCMPGTW:
    case LIBXSMM_X86_INSTR_VPCMPGTD:
    case LIBXSMM_X86_INSTR_VPCMPGTQ:
    case LIBXSMM_X86_INSTR_VPCMPESTRI:
    case LIBXSMM_X86_INSTR_VPCMPESTRM:
    case LIBXSMM_X86_INSTR_VPCMPISTRI:
    case LIBXSMM_X86_INSTR_VPCMPISTRM:
    case LIBXSMM_X86_INSTR_VCVTPS2PD:
    case LIBXSMM_X86_INSTR_VCVTPH2PS:
    case LIBXSMM_X86_INSTR_VCVTPS2PH:
    case LIBXSMM_X86_INSTR_VCVTDQ2PS:
    case LIBXSMM_X86_INSTR_VCVTPS2DQ:
    case LIBXSMM_X86_INSTR_VCVTPS2UDQ:
    case LIBXSMM_X86_INSTR_VPMOVDW:
    case LIBXSMM_X86_INSTR_VPMOVSXWD:
    case LIBXSMM_X86_INSTR_VPMOVDB:
    case LIBXSMM_X86_INSTR_VPMOVSDB:
    case LIBXSMM_X86_INSTR_VPMOVUSDB:
    case LIBXSMM_X86_INSTR_VPMOVZXWD:
    case LIBXSMM_X86_INSTR_VPMOVSXBD:
    case LIBXSMM_X86_INSTR_VPMOVZXBD:
    case LIBXSMM_X86_INSTR_VPMOVUSWB:
    case LIBXSMM_X86_INSTR_VPMOVSWB:
    case LIBXSMM_X86_INSTR_VPMOVWB:
    case LIBXSMM_X86_INSTR_VPSLLD_I:
    case LIBXSMM_X86_INSTR_VPSRAD_I:
    case LIBXSMM_X86_INSTR_VPSRLD_I:
    case LIBXSMM_X86_INSTR_VPSLLVW:
    case LIBXSMM_X86_INSTR_VPSLLVD:
    case LIBXSMM_X86_INSTR_VPSLLVQ:
    case LIBXSMM_X86_INSTR_VPSRAVW:
    case LIBXSMM_X86_INSTR_VPSRAVD:
    case LIBXSMM_X86_INSTR_VPSRAVQ:
    case LIBXSMM_X86_INSTR_VPSRLVW:
    case LIBXSMM_X86_INSTR_VPSRLVD:
    case LIBXSMM_X86_INSTR_VPSRLVQ:
    case LIBXSMM_X86_INSTR_VXORPD:
    case LIBXSMM_X86_INSTR_VADDPD:
    case LIBXSMM_X86_INSTR_VMULPD:
    case LIBXSMM_X86_INSTR_VSUBPD:
    case LIBXSMM_X86_INSTR_VDIVPD:
    case LIBXSMM_X86_INSTR_VMAXPD:
    case LIBXSMM_X86_INSTR_VADDSD:
    case LIBXSMM_X86_INSTR_VMULSD:
    case LIBXSMM_X86_INSTR_VSUBSD:
    case LIBXSMM_X86_INSTR_VXORPS:
    case LIBXSMM_X86_INSTR_VADDPS:
    case LIBXSMM_X86_INSTR_VMULPS:
    case LIBXSMM_X86_INSTR_VSUBPS:
    case LIBXSMM_X86_INSTR_VDIVPS:
    case LIBXSMM_X86_INSTR_VMAXPS:
    case LIBXSMM_X86_INSTR_VMULSS:
    case LIBXSMM_X86_INSTR_VADDSS:
    case LIBXSMM_X86_INSTR_VSUBSS:
    case LIBXSMM_X86_INSTR_VPXORD:
    case LIBXSMM_X86_INSTR_VPORD:
    case LIBXSMM_X86_INSTR_VPANDD:
    case LIBXSMM_X86_INSTR_VPANDQ:
    case LIBXSMM_X86_INSTR_VPADDQ:
    case LIBXSMM_X86_INSTR_VPADDB:
    case LIBXSMM_X86_INSTR_VPADDW:
    case LIBXSMM_X86_INSTR_VPADDD:
    case LIBXSMM_X86_INSTR_VPMADDWD:
    case LIBXSMM_X86_INSTR_VPMADDUBSW:
    case LIBXSMM_X86_INSTR_VPADDSW:
    case LIBXSMM_X86_INSTR_VPADDSB:
    case LIBXSMM_X86_INSTR_VPSUBD:
    case LIBXSMM_X86_INSTR_VPMAXSD:
    case LIBXSMM_X86_INSTR_VPMINSD:
    case LIBXSMM_X86_INSTR_V4FMADDPS:
    case LIBXSMM_X86_INSTR_V4FNMADDPS:
    case LIBXSMM_X86_INSTR_V4FMADDSS:
    case LIBXSMM_X86_INSTR_V4FNMADDSS:
    case LIBXSMM_X86_INSTR_VP4DPWSSDS:
    case LIBXSMM_X86_INSTR_VP4DPWSSD:
    case LIBXSMM_X86_INSTR_VPDPBUSD:
    case LIBXSMM_X86_INSTR_VPDPBUSDS:
    case LIBXSMM_X86_INSTR_VPDPWSSD:
    case LIBXSMM_X86_INSTR_VPDPWSSDS:
    case LIBXSMM_X86_INSTR_VDPBF16PS:
    case LIBXSMM_X86_INSTR_VCVTNEPS2BF16:
    case LIBXSMM_X86_INSTR_VCVTNE2PS2BF16:
    case LIBXSMM_X86_INSTR_VMOVDQU64_LD:
    case LIBXSMM_X86_INSTR_VMOVDQU64_ST:
    case LIBXSMM_X86_INSTR_VMOVD_LD:
    case LIBXSMM_X86_INSTR_VMOVQ_LD:
    case LIBXSMM_X86_INSTR_VMOVD_ST:
    case LIBXSMM_X86_INSTR_VMOVQ_ST:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: unexpected instruction number: %u\n", i_vec_instr);
      exit(-1);
  }

  /* check that we are not masking 'y' */
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512) && (i_mask_reg_number != 0) ) {
    fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: Masking is only available for AVX512!\n");
    exit(-1);
  }

  /* check for currently support archs in this encoder */
  if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
    fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: target error!\n");
    exit(-1);
  }

  /* select the code generator REX/VEX/EVEX */
  if ( (i_vec_instr >= 16777216) && (io_generated_code->code_type > 1) ) {
    unsigned int l_encoder; /* 2=EVEX, 1=VEX, 0=REX */
    unsigned int l_encoder_arch = 2;
    unsigned int l_encoder_instr = ((i_vec_instr >> 30) & 0x03);
    unsigned int l_reg_number_src1;
    unsigned int l_reg_number_dst = i_reg_number_dst;

    /* check if we have enough code buffer space left */
    if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* determine encoder */
    if ( io_generated_code->arch < LIBXSMM_X86_AVX512) {
      l_encoder_arch = 1;
    } else if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
      l_encoder_arch = 0;
    }
    if ( (l_encoder_arch == 2) && ((l_encoder_instr == 3) || (l_encoder_instr == 0)) ) {
      l_encoder = 2;
    } else if ( (l_encoder_arch >= 1) && ((l_encoder_instr == 1) || (l_encoder_instr == 0)) ) {
      l_encoder = 1;
    } else {
      l_encoder = 0;
    }

    /* check that we have an UNDEF for 2 src operands */
    if ( ((i_vec_instr >> 28) & 3) == 2 ) {
      if ( i_reg_number_src1 != LIBXSMM_X86_VEC_REG_UNDEF ) {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: In case of a 1 src operand instruction (%u), i_reg_number_src1 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        exit(-1);
      }
      l_reg_number_src1 = 0;
    } else {
      l_reg_number_src1 = i_reg_number_src1;
    }

    /* check that we have an UNDEF for both vec reg operands */
    if ( ((i_vec_instr >> 28) & 3) == 1 ) {
      if ( (i_reg_number_src1 != LIBXSMM_X86_VEC_REG_UNDEF) || (i_reg_number_dst != LIBXSMM_X86_VEC_REG_UNDEF) ) {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: In case of a 0 src operand instruction (%u), i_reg_number_src1 and i_reg_number_dst needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        exit(-1);
      }
      l_reg_number_src1 = 0;
      l_reg_number_dst = 0;
    }

    /* check if we have op-code extension in modrm/reg */
    if ( ((i_vec_instr >> 24) & 0x04 ) == 0x04 ) {
      if ( ((i_vec_instr >> 28) & 0x3) == 0x2 ) {
        l_reg_number_src1 = l_reg_number_dst;
        l_reg_number_dst = ((i_vec_instr >> 20) & 0x07);
      } else {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: In case of a op-code modrm/reg extended instruction (%u), i_reg_number_src1 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        exit(-1);
      }
    }

    /* on Knights platfrom, attempt to fallback to VEX for ymm and xmm VL,
     * will error out in the encoder if instruction doesn't have VEX encoding
     * Core will always take AVX512VL route */
    if ( ( (io_generated_code->arch == LIBXSMM_X86_AVX512_MIC) || (io_generated_code->arch == LIBXSMM_X86_AVX512_KNM) ) &&
         ( (i_vector_name == 'x') || (i_vector_name == 'y') ) && (l_encoder == 2) ) {
      l_encoder = 1;
    }

    /* encode main instruction */
    if ( l_encoder == 2 ) {
      libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;

      /* set simd name */
      switch(i_vector_name) {
        case 'x':
          l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
          break;
        case 'y':
          l_simd_name = LIBXSMM_X86_SIMD_NAME_YMM;
          break;
        case 'z':
          l_simd_name = LIBXSMM_X86_SIMD_NAME_ZMM;
          break;
        default:
          fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: unsupported vlen: %c\n", i_vector_name);
          break;
      }

      libxsmm_x86_instruction_evex_compute_2reg_mem( io_generated_code, i_vec_instr,
            i_use_broadcast, i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, l_simd_name,
            l_reg_number_src1, l_reg_number_dst, i_mask_reg_number, i_mask_rnd_exp_cntl );
    } else if ( l_encoder == 1 ) {
      libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;

      /* set simd name */
      switch(i_vector_name) {
        case 'x':
          l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;
          break;
        case 'y':
          l_simd_name = LIBXSMM_X86_SIMD_NAME_YMM;
          break;
        default:
          fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: unsupported vlen: %c\n", i_vector_name);
          break;
      }

      libxsmm_x86_instruction_vex_compute_2reg_mem( io_generated_code, i_vec_instr,
            i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, l_simd_name,
            l_reg_number_src1, l_reg_number_dst );
    } else {
      fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: No REX encoder available!\n");
      exit(-1);
    }

    /* add imm if needed */
    if ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) {
      if ( i_imm8 != LIBXSMM_X86_IMM_UNDEF ) {
        unsigned char* code = (unsigned char *) io_generated_code->generated_code;
        code[io_generated_code->code_size++] = (unsigned char)i_imm8;
      } else {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: imm8 required by instr, but LIBXSMM_X86_IMM_UNDEF was provided!\n");
        exit(-1);
      }
    }
  } else {
    fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: GENERAL ERROR!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_2reg( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_vec_instr,
                                                   const char              i_vector_name,
                                                   const unsigned int      i_gp_reg_base,
                                                   const unsigned int      i_gp_reg_idx,
                                                   const unsigned int      i_scale,
                                                   const int               i_displacement,
                                                   const unsigned int      i_use_broadcast,
                                                   const unsigned int      i_reg_number_src1,
                                                   const unsigned int      i_reg_number_dst ) {
  libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code, i_vec_instr, i_vector_name,
                                                          i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_use_broadcast,
                                                          i_reg_number_src1, i_reg_number_dst,
                                                          0, 0, LIBXSMM_X86_IMM_UNDEF );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_2reg_mask( libxsmm_generated_code* io_generated_code,
                                                        const unsigned int      i_vec_instr,
                                                        const char              i_vector_name,
                                                        const unsigned int      i_gp_reg_base,
                                                        const unsigned int      i_gp_reg_idx,
                                                        const unsigned int      i_scale,
                                                        const int               i_displacement,
                                                        const unsigned int      i_use_broadcast,
                                                        const unsigned int      i_reg_number_src1,
                                                        const unsigned int      i_reg_number_dst,
                                                        const unsigned int      i_mask_reg_number,
                                                        const unsigned int      i_mask_rnd_exp_cntl ) {
  libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code, i_vec_instr, i_vector_name,
                                                          i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_use_broadcast,
                                                          i_reg_number_src1, i_reg_number_dst,
                                                          i_mask_reg_number, i_mask_rnd_exp_cntl, LIBXSMM_X86_IMM_UNDEF );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_2reg_imm8( libxsmm_generated_code* io_generated_code,
                                                        const unsigned int      i_vec_instr,
                                                        const char              i_vector_name,
                                                        const unsigned int      i_gp_reg_base,
                                                        const unsigned int      i_gp_reg_idx,
                                                        const unsigned int      i_scale,
                                                        const int               i_displacement,
                                                        const unsigned int      i_use_broadcast,
                                                        const unsigned int      i_reg_number_src1,
                                                        const unsigned int      i_reg_number_dst,
                                                        const unsigned short    i_imm8 ) {
  libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code, i_vec_instr, i_vector_name,
                                                          i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_use_broadcast,
                                                          i_reg_number_src1, i_reg_number_dst,
                                                          0, 0, i_imm8 );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_1reg_mask_imm8( libxsmm_generated_code* io_generated_code,
                                                             const unsigned int      i_vec_instr,
                                                             const char              i_vector_name,
                                                             const unsigned int      i_gp_reg_base,
                                                             const unsigned int      i_gp_reg_idx,
                                                             const unsigned int      i_scale,
                                                             const int               i_displacement,
                                                             const unsigned int      i_use_broadcast,
                                                             const unsigned int      i_reg_number_dst,
                                                             const unsigned int      i_mask_reg_number,
                                                             const unsigned int      i_mask_rnd_exp_cntl,
                                                             const unsigned short    i_imm8 ) {
  libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code, i_vec_instr, i_vector_name,
                                                          i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_use_broadcast,
                                                          LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
                                                          i_mask_reg_number, i_mask_rnd_exp_cntl, i_imm8 );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_1reg( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_vec_instr,
                                                   const char              i_vector_name,
                                                   const unsigned int      i_gp_reg_base,
                                                   const unsigned int      i_gp_reg_idx,
                                                   const unsigned int      i_scale,
                                                   const int               i_displacement,
                                                   const unsigned int      i_use_broadcast,
                                                   const unsigned int      i_reg_number_dst ) {
  libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code, i_vec_instr, i_vector_name,
                                                          i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_use_broadcast,
                                                          LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
                                                          0, 0, LIBXSMM_X86_IMM_UNDEF );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_1reg_mask( libxsmm_generated_code* io_generated_code,
                                                        const unsigned int      i_vec_instr,
                                                        const char              i_vector_name,
                                                        const unsigned int      i_gp_reg_base,
                                                        const unsigned int      i_gp_reg_idx,
                                                        const unsigned int      i_scale,
                                                        const int               i_displacement,
                                                        const unsigned int      i_use_broadcast,
                                                        const unsigned int      i_reg_number_dst,
                                                        const unsigned int      i_mask_reg_number,
                                                        const unsigned int      i_mask_rnd_exp_cntl ) {
  libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code, i_vec_instr, i_vector_name,
                                                          i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_use_broadcast,
                                                          LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
                                                          i_mask_reg_number, i_mask_rnd_exp_cntl, LIBXSMM_X86_IMM_UNDEF );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_1reg_imm8( libxsmm_generated_code* io_generated_code,
                                                        const unsigned int      i_vec_instr,
                                                        const char              i_vector_name,
                                                        const unsigned int      i_gp_reg_base,
                                                        const unsigned int      i_gp_reg_idx,
                                                        const unsigned int      i_scale,
                                                        const int               i_displacement,
                                                        const unsigned int      i_use_broadcast,
                                                        const unsigned int      i_reg_number_dst,
                                                        const unsigned short    i_imm8 ) {
  libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code, i_vec_instr, i_vector_name,
                                                          i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_use_broadcast,
                                                          LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
                                                          0, 0, i_imm8 );
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
  switch ( i_vec_instr ) {
    case LIBXSMM_X86_INSTR_VXORPD:
    case LIBXSMM_X86_INSTR_VMULPD:
    case LIBXSMM_X86_INSTR_VPERMW:
    case LIBXSMM_X86_INSTR_VPERMD:
    case LIBXSMM_X86_INSTR_VUNPCKLPD:
    case LIBXSMM_X86_INSTR_VUNPCKLPS:
    case LIBXSMM_X86_INSTR_VUNPCKHPD:
    case LIBXSMM_X86_INSTR_VUNPCKHPS:
    case LIBXSMM_X86_INSTR_VADDPD:
    case LIBXSMM_X86_INSTR_VDIVPD:
    case LIBXSMM_X86_INSTR_VDPBF16PS:
    case LIBXSMM_X86_INSTR_VDIVPS:
    case LIBXSMM_X86_INSTR_VPANDD:
    case LIBXSMM_X86_INSTR_VPANDQ:
    case LIBXSMM_X86_INSTR_VMAXPD:
    case LIBXSMM_X86_INSTR_VMAXPS:
    case LIBXSMM_X86_INSTR_VCVTDQ2PS:
    case LIBXSMM_X86_INSTR_VCVTPS2PD:
    case LIBXSMM_X86_INSTR_VRCP14PS:
    case LIBXSMM_X86_INSTR_VMOVDQU64:
    case LIBXSMM_X86_INSTR_VPMAXSD:
    case LIBXSMM_X86_INSTR_VPMINSD:
    case LIBXSMM_X86_INSTR_VSUBPD:
    case LIBXSMM_X86_INSTR_VPADDD:
    case LIBXSMM_X86_INSTR_VPADDQ:
    case LIBXSMM_X86_INSTR_VPADDW:
    case LIBXSMM_X86_INSTR_VPADDB:
    case LIBXSMM_X86_INSTR_VPMADDWD:
    case LIBXSMM_X86_INSTR_VPMADDUBSW:
    case LIBXSMM_X86_INSTR_VPADDSW:
    case LIBXSMM_X86_INSTR_VPADDSB:
    case LIBXSMM_X86_INSTR_VFMADD231PD:
    case LIBXSMM_X86_INSTR_VFMADD213PD:
    case LIBXSMM_X86_INSTR_VFMADD132PD:
    case LIBXSMM_X86_INSTR_VFMSUB231PD:
    case LIBXSMM_X86_INSTR_VFMSUB213PD:
    case LIBXSMM_X86_INSTR_VFMSUB132PD:
    case LIBXSMM_X86_INSTR_VFNMADD231PD:
    case LIBXSMM_X86_INSTR_VFNMADD213PD:
    case LIBXSMM_X86_INSTR_VFNMADD132PD:
    case LIBXSMM_X86_INSTR_VFNMSUB231PD:
    case LIBXSMM_X86_INSTR_VFNMSUB213PD:
    case LIBXSMM_X86_INSTR_VFNMSUB132PD:
    case LIBXSMM_X86_INSTR_VFMADD231SD:
    case LIBXSMM_X86_INSTR_VFMADD213SD:
    case LIBXSMM_X86_INSTR_VFMADD132SD:
    case LIBXSMM_X86_INSTR_VFMADD213SS:
    case LIBXSMM_X86_INSTR_VFMADD132SS:
    case LIBXSMM_X86_INSTR_VFMSUB213SS:
    case LIBXSMM_X86_INSTR_VFMSUB132SS:
    case LIBXSMM_X86_INSTR_VFNMADD213SS:
    case LIBXSMM_X86_INSTR_VFNMADD132SS:
    case LIBXSMM_X86_INSTR_VFNMSUB213SS:
    case LIBXSMM_X86_INSTR_VFNMSUB132SS:
    case LIBXSMM_X86_INSTR_VFNMADD213SD:
    case LIBXSMM_X86_INSTR_VFNMADD132SD:
    case LIBXSMM_X86_INSTR_VFMSUB213SD:
    case LIBXSMM_X86_INSTR_VFMSUB132SD:
    case LIBXSMM_X86_INSTR_VFNMSUB213SD:
    case LIBXSMM_X86_INSTR_VFNMSUB132SD:
    case LIBXSMM_X86_INSTR_VFMSUB231SD:
    case LIBXSMM_X86_INSTR_VFNMADD231SD:
    case LIBXSMM_X86_INSTR_VFNMSUB231SD:
    case LIBXSMM_X86_INSTR_VFMADD231PS:
    case LIBXSMM_X86_INSTR_VFMADD213PS:
    case LIBXSMM_X86_INSTR_VFMADD132PS:
    case LIBXSMM_X86_INSTR_VFNMADD213PS:
    case LIBXSMM_X86_INSTR_VFNMADD132PS:
    case LIBXSMM_X86_INSTR_VFNMSUB213PS:
    case LIBXSMM_X86_INSTR_VFNMSUB132PS:
    case LIBXSMM_X86_INSTR_VFMSUB213PS:
    case LIBXSMM_X86_INSTR_VFMSUB132PS:
    case LIBXSMM_X86_INSTR_VFMSUB231PS:
    case LIBXSMM_X86_INSTR_VFNMADD231PS:
    case LIBXSMM_X86_INSTR_VFNMSUB231PS:
    case LIBXSMM_X86_INSTR_VFMADD231SS:
    case LIBXSMM_X86_INSTR_VFMSUB231SS:
    case LIBXSMM_X86_INSTR_VFNMADD231SS:
    case LIBXSMM_X86_INSTR_VFNMSUB231SS:
    case LIBXSMM_X86_INSTR_VMULSD:
    case LIBXSMM_X86_INSTR_VADDSD:
    case LIBXSMM_X86_INSTR_VSUBSD:
    case LIBXSMM_X86_INSTR_VXORPS:
    case LIBXSMM_X86_INSTR_VMULPS:
    case LIBXSMM_X86_INSTR_VADDPS:
    case LIBXSMM_X86_INSTR_VSUBPS:
    case LIBXSMM_X86_INSTR_VPSRAVD:
    case LIBXSMM_X86_INSTR_VMULSS:
    case LIBXSMM_X86_INSTR_VADDSS:
    case LIBXSMM_X86_INSTR_VSUBSS:
    case LIBXSMM_X86_INSTR_VPERMT2W:
    case LIBXSMM_X86_INSTR_VPXORD:
    case LIBXSMM_X86_INSTR_VPORD:
    case LIBXSMM_X86_INSTR_VPDPWSSD:
    case LIBXSMM_X86_INSTR_VPDPWSSDS:
    case LIBXSMM_X86_INSTR_VPDPBUSD:
    case LIBXSMM_X86_INSTR_VPDPBUSDS:
    case LIBXSMM_X86_INSTR_MOVAPD:
    case LIBXSMM_X86_INSTR_MOVUPD:
    case LIBXSMM_X86_INSTR_MOVAPS:
    case LIBXSMM_X86_INSTR_MOVUPS:
    case LIBXSMM_X86_INSTR_MOVSD:
    case LIBXSMM_X86_INSTR_MOVSS:
    case LIBXSMM_X86_INSTR_MOVDDUP:
    case LIBXSMM_X86_INSTR_XORPD:
    case LIBXSMM_X86_INSTR_XORPS:
    case LIBXSMM_X86_INSTR_MULPD:
    case LIBXSMM_X86_INSTR_MULPS:
    case LIBXSMM_X86_INSTR_ADDPD:
    case LIBXSMM_X86_INSTR_ADDPS:
    case LIBXSMM_X86_INSTR_SUBPD:
    case LIBXSMM_X86_INSTR_SUBPS:
    case LIBXSMM_X86_INSTR_MULSD:
    case LIBXSMM_X86_INSTR_MULSS:
    case LIBXSMM_X86_INSTR_ADDSD:
    case LIBXSMM_X86_INSTR_ADDSS:
    case LIBXSMM_X86_INSTR_SUBSD:
    case LIBXSMM_X86_INSTR_SUBSS:
      break;
    default:
      fprintf(stderr, "libxsmm_instruction_vec_compute_reg: Unknown instruction type: %u\n", i_vec_instr);
      exit(-1);
  }

  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) &&
       (io_generated_code->code_type > 1 ) ) {
    libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code,
                                                             i_vec_instr, i_vector_name,
                                                             i_vec_reg_number_0, i_vec_reg_number_1, i_vec_reg_number_2,
                                                             0, 0, 0, 0 );
  } else if ( (io_generated_code->arch < LIBXSMM_X86_AVX) &&
              (io_generated_code->code_type > 1 ) ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int l_third=0;
    int l_reg0, l_reg1;
    int l_vreg0   = i_vec_reg_number_0;
    int l_vreg1   = i_vec_reg_number_1;
    int l_insert_extra_byte = 0;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }

    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_MOVAPD:
          l_insert_extra_byte = 0x66;
          l_third = 0x18;
          break;
       case LIBXSMM_X86_INSTR_MOVUPD:
          l_insert_extra_byte = 0x66;
          break;
       case LIBXSMM_X86_INSTR_MOVAPS:
          l_third = 0x18;
          break;
       case LIBXSMM_X86_INSTR_MOVUPS:
          break;
       case LIBXSMM_X86_INSTR_MOVSD:
          l_insert_extra_byte = 0xF2;
          break;
       case LIBXSMM_X86_INSTR_MOVSS:
          l_insert_extra_byte = 0xF3;
          break;
       case LIBXSMM_X86_INSTR_MOVDDUP:
          l_third = 2;
          l_insert_extra_byte = 0xF2;
          break;
       case LIBXSMM_X86_INSTR_XORPD:
          l_insert_extra_byte = 0x66;
          l_third = 0x47;
          break;
       case LIBXSMM_X86_INSTR_XORPS:
          l_third = 0x47;
          break;
       case LIBXSMM_X86_INSTR_MULPD:
          l_insert_extra_byte = 0x66;
          l_third = 0x49;
          break;
       case LIBXSMM_X86_INSTR_MULPS:
          l_third = 0x49;
          break;
       case LIBXSMM_X86_INSTR_ADDPD:
          l_insert_extra_byte = 0x66;
          l_third = 0x48;
          break;
       case LIBXSMM_X86_INSTR_ADDPS:
          l_third = 0x48;
          break;
       case LIBXSMM_X86_INSTR_SUBPD:
          l_insert_extra_byte = 0x66;
          l_third = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_SUBPS:
          l_third = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_MULSD:
          l_insert_extra_byte = 0xF2;
          l_third = 0x49;
          break;
       case LIBXSMM_X86_INSTR_MULSS:
          l_insert_extra_byte = 0xF3;
          l_third = 0x49;
          break;
       case LIBXSMM_X86_INSTR_ADDSD:
          l_insert_extra_byte = 0xF2;
          l_third = 0x48;
          break;
       case LIBXSMM_X86_INSTR_ADDSS:
          l_insert_extra_byte = 0xF3;
          l_third = 0x48;
          break;
       case LIBXSMM_X86_INSTR_SUBSD:
          l_insert_extra_byte = 0xF2;
          l_third = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_SUBSS:
          l_insert_extra_byte = 0xF3;
          l_third = 0x4c;
          break;
       default:
          fprintf(stderr, "libxsmm_instruction_vec_compute_reg: Unknown instruction type: %u\n", i_vec_instr);
          exit(-1);
    }
    l_reg0 = l_vreg0 % 8;
    l_reg1 = l_vreg1 % 8;
    {
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
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name, 15 );

    /* build vXYZpd/ps/sd/ss instruction pure register use*/
    if ( i_instruction_set > LIBXSMM_X86_SSE42 ) {
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
                                              const unsigned int      i_vec_reg_number_1 )
{
  switch ( i_vec_instr ) {
    case LIBXSMM_X86_INSTR_VXORPD:
    case LIBXSMM_X86_INSTR_VMULPD:
    case LIBXSMM_X86_INSTR_VADDPD:
    case LIBXSMM_X86_INSTR_VPANDD:
    case LIBXSMM_X86_INSTR_VSUBPD:
    case LIBXSMM_X86_INSTR_VMAXPD:
    case LIBXSMM_X86_INSTR_VMAXPS:
    case LIBXSMM_X86_INSTR_VPERMW:
    case LIBXSMM_X86_INSTR_VPERMD:
    case LIBXSMM_X86_INSTR_VFMADD231PD:
    case LIBXSMM_X86_INSTR_VFMADD213PD:
    case LIBXSMM_X86_INSTR_VFMADD132PD:
    case LIBXSMM_X86_INSTR_VFMSUB231PD:
    case LIBXSMM_X86_INSTR_VFMSUB213PD:
    case LIBXSMM_X86_INSTR_VFMSUB132PD:
    case LIBXSMM_X86_INSTR_VFNMADD231PD:
    case LIBXSMM_X86_INSTR_VFNMADD213PD:
    case LIBXSMM_X86_INSTR_VFNMADD132PD:
    case LIBXSMM_X86_INSTR_VFNMSUB231PD:
    case LIBXSMM_X86_INSTR_VFNMSUB213PD:
    case LIBXSMM_X86_INSTR_VFNMSUB132PD:
    case LIBXSMM_X86_INSTR_VFMADD231SD:
    case LIBXSMM_X86_INSTR_VFMADD213SD:
    case LIBXSMM_X86_INSTR_VFMADD132SD:
    case LIBXSMM_X86_INSTR_VFMSUB231SD:
    case LIBXSMM_X86_INSTR_VFMSUB213SD:
    case LIBXSMM_X86_INSTR_VFMSUB132SD:
    case LIBXSMM_X86_INSTR_VFNMADD231SD:
    case LIBXSMM_X86_INSTR_VFNMADD213SD:
    case LIBXSMM_X86_INSTR_VFNMADD132SD:
    case LIBXSMM_X86_INSTR_VFNMSUB231SD:
    case LIBXSMM_X86_INSTR_VFNMSUB213SD:
    case LIBXSMM_X86_INSTR_VFNMSUB132SD:
    case LIBXSMM_X86_INSTR_VFMADD231PS:
    case LIBXSMM_X86_INSTR_VFMADD213PS:
    case LIBXSMM_X86_INSTR_VFMADD132PS:
    case LIBXSMM_X86_INSTR_VFMSUB231PS:
    case LIBXSMM_X86_INSTR_VFMSUB213PS:
    case LIBXSMM_X86_INSTR_VFMSUB132PS:
    case LIBXSMM_X86_INSTR_VFNMADD231PS:
    case LIBXSMM_X86_INSTR_VFNMADD213PS:
    case LIBXSMM_X86_INSTR_VFNMADD132PS:
    case LIBXSMM_X86_INSTR_VFNMSUB231PS:
    case LIBXSMM_X86_INSTR_VFNMSUB213PS:
    case LIBXSMM_X86_INSTR_VFNMSUB132PS:
    case LIBXSMM_X86_INSTR_VFMADD231SS:
    case LIBXSMM_X86_INSTR_VFMADD213SS:
    case LIBXSMM_X86_INSTR_VFMADD132SS:
    case LIBXSMM_X86_INSTR_VFMSUB231SS:
    case LIBXSMM_X86_INSTR_VFMSUB213SS:
    case LIBXSMM_X86_INSTR_VFMSUB132SS:
    case LIBXSMM_X86_INSTR_VFNMADD231SS:
    case LIBXSMM_X86_INSTR_VFNMADD213SS:
    case LIBXSMM_X86_INSTR_VFNMADD132SS:
    case LIBXSMM_X86_INSTR_VFNMSUB231SS:
    case LIBXSMM_X86_INSTR_VFNMSUB213SS:
    case LIBXSMM_X86_INSTR_VFNMSUB132SS:
    case LIBXSMM_X86_INSTR_VMULSD:
    case LIBXSMM_X86_INSTR_VADDSD:
    case LIBXSMM_X86_INSTR_VSUBSD:
    case LIBXSMM_X86_INSTR_VPMOVDW:
    case LIBXSMM_X86_INSTR_VPMOVSXWD:
    case LIBXSMM_X86_INSTR_VPMOVZXWD:
    case LIBXSMM_X86_INSTR_VPMOVSXBD:
    case LIBXSMM_X86_INSTR_VPMOVZXBD:
    case LIBXSMM_X86_INSTR_VXORPS:
    case LIBXSMM_X86_INSTR_VMULPS:
    case LIBXSMM_X86_INSTR_VDPBF16PS:
    case LIBXSMM_X86_INSTR_VCVTNE2PS2BF16:
    case LIBXSMM_X86_INSTR_VADDPS:
    case LIBXSMM_X86_INSTR_VSUBPS:
    case LIBXSMM_X86_INSTR_VMULSS:
    case LIBXSMM_X86_INSTR_VADDSS:
    case LIBXSMM_X86_INSTR_VSUBSS:
    case LIBXSMM_X86_INSTR_VPXORD:
    case LIBXSMM_X86_INSTR_VPORD:
    case LIBXSMM_X86_INSTR_VPADDD:
    case LIBXSMM_X86_INSTR_MOVAPD:
    case LIBXSMM_X86_INSTR_MOVUPD:
    case LIBXSMM_X86_INSTR_MOVAPS:
    case LIBXSMM_X86_INSTR_MOVUPS:
    case LIBXSMM_X86_INSTR_MOVSD:
    case LIBXSMM_X86_INSTR_MOVSS:
    case LIBXSMM_X86_INSTR_MOVDDUP:
    case LIBXSMM_X86_INSTR_XORPD:
    case LIBXSMM_X86_INSTR_XORPS:
    case LIBXSMM_X86_INSTR_MULPD:
    case LIBXSMM_X86_INSTR_MULSS:
    case LIBXSMM_X86_INSTR_MULPS:
    case LIBXSMM_X86_INSTR_ADDPD:
    case LIBXSMM_X86_INSTR_ADDSS:
    case LIBXSMM_X86_INSTR_ADDPS:
    case LIBXSMM_X86_INSTR_ADDSD:
    case LIBXSMM_X86_INSTR_SUBPD:
    case LIBXSMM_X86_INSTR_SUBSS:
    case LIBXSMM_X86_INSTR_SUBPS:
    case LIBXSMM_X86_INSTR_SUBSD:
    case LIBXSMM_X86_INSTR_MULSD:
      break;
    default:
      fprintf(stderr, "libxsmm_instruction_vec_compute_mem: Unknown instruction type: %u\n", i_vec_instr);
      exit(-1);
      break;
  }

  if ( (i_instruction_set < LIBXSMM_X86_AVX512)  &&
       (i_use_broadcast != 0) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_NO_AVX512_BCAST );
    return;
  }

  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) &&
       (io_generated_code->code_type > 1 ) ) {
    libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code, i_vec_instr, i_vector_name,
                                                            i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_use_broadcast,
                                                            i_vec_reg_number_0, i_vec_reg_number_1, 0, 0, 0 );

  } else if ( (io_generated_code->arch < LIBXSMM_X86_AVX) &&
              (io_generated_code->code_type > 1 ) ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int l_fpadj=0;
    int l_forced_offset=0;
    int l_scaleadj=0;
    int l_insert_extra_byte = 0;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }
    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_MOVAPD:
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x18;
          break;
       case LIBXSMM_X86_INSTR_MOVUPD:
          l_insert_extra_byte = 0x66;
          break;
       case LIBXSMM_X86_INSTR_MOVAPS:
          l_fpadj = 0x18;
          break;
       case LIBXSMM_X86_INSTR_MOVUPS:
          break;
       case LIBXSMM_X86_INSTR_MOVSD:
          l_insert_extra_byte = 0xF2;
          break;
       case LIBXSMM_X86_INSTR_MOVSS:
          l_insert_extra_byte = 0xF3;
          break;
       case LIBXSMM_X86_INSTR_MOVDDUP:
          l_insert_extra_byte = 0xF2;
          l_fpadj = 2;
          break;
       case LIBXSMM_X86_INSTR_XORPD:
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x47;
          break;
       case LIBXSMM_X86_INSTR_XORPS:
          l_fpadj = 0x47;
          break;
       case LIBXSMM_X86_INSTR_MULPD:
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x49;
          break;
       case LIBXSMM_X86_INSTR_MULSS:
          l_insert_extra_byte = 0xF3;
          l_fpadj = 0x49;
          break;
       case LIBXSMM_X86_INSTR_MULPS:
          l_fpadj = 0x49;
          break;
       case LIBXSMM_X86_INSTR_ADDPD:
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x48;
          break;
       case LIBXSMM_X86_INSTR_ADDSS:
          l_insert_extra_byte = 0xF3;
          l_fpadj = 0x48;
          break;
       case LIBXSMM_X86_INSTR_ADDPS:
          l_fpadj = 0x48;
          break;
       case LIBXSMM_X86_INSTR_ADDSD:
          l_insert_extra_byte = 0xF2;
          l_fpadj = 0x48;
          break;
       case LIBXSMM_X86_INSTR_SUBPD:
          l_insert_extra_byte = 0x66;
          l_fpadj = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_SUBSS:
          l_insert_extra_byte = 0xF3;
          l_fpadj = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_SUBPS:
          l_fpadj = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_SUBSD:
          l_insert_extra_byte = 0xF2;
          l_fpadj = 0x4c;
          break;
       case LIBXSMM_X86_INSTR_MULSD:
          l_insert_extra_byte = 0xF2;
          l_fpadj = 0x49;
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
            fprintf(stderr, "libxsmm_instruction_vec_compute_mem: cannot handle i_scale=%u parameter\n", i_scale);
            exit(-1);
       }
    }
    {
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
void libxsmm_x86_instruction_vec_shuffle_sse_reg( libxsmm_generated_code* io_generated_code,
                                                  const unsigned int      i_vec_instr,
                                                  const char              i_vector_name,
                                                  const unsigned int      i_vec_reg_number_0,
                                                  const unsigned int      i_vec_reg_number_1,
                                                  const unsigned int      i_shuffle_operand ) {
  switch ( i_vec_instr ) {
    case LIBXSMM_X86_INSTR_SHUFPS:
    case LIBXSMM_X86_INSTR_SHUFPD:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_sse_reg: Unknown instruction type: %u\n", i_vec_instr);
      exit(-1);
      break;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int l_vecval0 = i_vec_reg_number_0 % 8;
    int l_vecgrp0 = i_vec_reg_number_0 / 8;
    int l_vecval1 = i_vec_reg_number_1 % 8;
    int l_vecgrp1 = i_vec_reg_number_1 / 8;
    int l_extra_byte = 0;

    if ( l_maxsize - i < 20 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }

    switch ( i_vec_instr ) {
       case LIBXSMM_X86_INSTR_SHUFPS:
          if ( (i_vector_name!='x') && (i_vector_name!='X') )
          {
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_sse_reg: SHUFPS only works for xmm\n");
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
             fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_sse_reg: SHUFPD only works for xmm\n");
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
       default:
          fprintf(stderr, "libxsmm_x86_instruction_vec_shuffle_sse_reg doesn't yet do this instruction\n");
          exit(-1);
          break;
    }

    /* Every instruction in this group has 1 byte at the end with the operand */
    buf[i++] = (unsigned char)(i_shuffle_operand);

    io_generated_code->code_size = i;
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[16];
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name, 15 );

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s $%u, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_shuffle_operand, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%u, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_shuffle_operand, i_vector_name, i_vec_reg_number_0, i_vector_name, i_vec_reg_number_1 );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vex_evex_mask_mov( libxsmm_generated_code* io_generated_code,
                                                const unsigned int      i_vmove_instr,
                                                const unsigned int      i_gp_reg_base,
                                                const unsigned int      i_reg_idx,
                                                const unsigned int      i_scale,
                                                const int               i_displacement,
                                                const char              i_vector_name,
                                                const unsigned int      i_vec_reg_number_0,
                                                const unsigned int      i_use_masking,
                                                const unsigned int      i_mask_reg_number,
                                                const unsigned int      i_is_store ) {
  if ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) {
    if ( i_use_masking != 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_vmove_instr,
                                        i_gp_reg_base, i_reg_idx, i_scale, i_displacement,
                                        i_vector_name, i_vec_reg_number_0, i_mask_reg_number, (i_is_store != 0) ? 0 : 1, i_is_store );
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_vmove_instr,
                                        i_gp_reg_base, i_reg_idx, i_scale, i_displacement,
                                        i_vector_name, i_vec_reg_number_0, 0, (i_is_store != 0) ? 0 : 1, i_is_store );
    }
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) && (io_generated_code->arch < LIBXSMM_X86_AVX512) ) {
    if ( i_use_masking != 0 ) {
      libxsmm_x86_instruction_vec_mask_move( io_generated_code, i_vmove_instr,
                                             i_gp_reg_base, i_reg_idx, i_scale, i_displacement,
                                             i_vector_name, i_vec_reg_number_0, i_mask_reg_number, i_is_store );
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_vmove_instr,
                                        i_gp_reg_base, i_reg_idx, i_scale, i_displacement,
                                        i_vector_name, i_vec_reg_number_0, 0, 1, i_is_store );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_prefetch( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_prefetch_instr,
                                       const unsigned int      i_gp_reg_base,
                                       const unsigned int      i_gp_reg_idx,
                                       const unsigned int      i_scale,
                                       const int               i_displacement ) {
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
    int l_opcode = 0;
    int l_havepf = 0;

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
       case LIBXSMM_X86_INSTR_CLDEMOTE:
          l_opcode = 0x4;
          l_instype -= 16;
          break;
       case LIBXSMM_X86_INSTR_CLFLUSHOPT:
          l_havepf = 0x66;
          l_opcode = 0x96;
          l_instype += 0x28;
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

    if ( l_havepf ) {
      buf[i++] = (unsigned char)l_havepf;
      ++l_place1;
    }

    if ( l_gp8 || l_ix8 )
    {
      if (l_gp8) l_sse_preamble += 1;
      if (l_ix8) l_sse_preamble += 2;
      buf[i++] = (unsigned char)l_sse_preamble;
      ++l_place1;
    }

    if (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF ){
      LIBXSMM_ASSERT(i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF);
      buf[i++] = 0x0f;
      buf[i++] = (unsigned char)(0x18 + l_opcode);
      buf[i++] = (unsigned char)(0x10 + l_instype + l_regbas0);
      if ( l_regbas0 == 4 ) buf[i++]=0x24;
    } else {
      const int l_regidx = i_gp_reg_idx % 8;
      int l_sca = 0;
      if (i_scale == 2) l_sca = 0x40;
      else if (i_scale == 4) l_sca = 0x80;
      else if (i_scale == 8) l_sca = 0xc0;
      buf[i++] = 0x0f;
      buf[i++] = (unsigned char)(0x18 + l_opcode);
      buf[i++] = (unsigned char)(0x14 + l_instype);
      buf[i++] = (unsigned char)(0x00 + l_sca + l_regbas0 + l_regidx*8);
    }

    if ( ( l_regbas0 == 5) && (i_displacement==0) )
    {
      /* Registers like rbp/r13 when you have a displacement of 0, we need
       * force the single byte of zero to appear.
       */
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
    char l_gp_reg_idx_name[4];
    char l_instr_name[16];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base_name, 3 );
    libxsmm_get_x86_instr_name( i_prefetch_instr, l_instr_name, 15 );

    if ( io_generated_code->code_type == 0 ) {
      if (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s)\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name );
      } else {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx_name, 3 );
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%%s,%u)\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base_name, l_gp_reg_idx_name, i_scale );
      }
    } else {
      if (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s)\n", l_instr_name, i_displacement, l_gp_reg_base_name );
      } else {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx_name, 3 );
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u)\n", l_instr_name, i_displacement, l_gp_reg_base_name, l_gp_reg_idx_name, i_scale );
      }
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
     int l_force_rex = 0;
     int l_sixsix_pre = 0;

     switch ( i_alu_instr ) {
       case LIBXSMM_X86_INSTR_MOVSLQ:
          l_force_rex = 1;
          if ( i_is_store == 1 )
          {
             fprintf(stderr, "libxsmm_instruction_alu_mem: only use LIBXSMM_X86_INSTR_MOVSLQ with loads\n");
             exit(-1);
          }
          break;
       case LIBXSMM_X86_INSTR_MOVQ:
          l_force_rex = 1;
          if ( i_is_store == 1 )
          {
             l_inst = 0x26;
          } else {
             l_inst = 0x28;
          }
          break;
       case LIBXSMM_X86_INSTR_LEAQ:
          l_force_rex = 1;
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
       case LIBXSMM_X86_INSTR_MOVW:
          l_sixsix_pre = 1;
          if ( i_is_store == 1 )
          {
             l_inst = 0x26;
          } else {
             l_inst = 0x28;
          }
          l_base = -8;
          break;
       case LIBXSMM_X86_INSTR_MOVB:
          if ( i_is_store == 1 )
          {
             l_inst = 0x25;
          } else {
             l_inst = 0x27;
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

     if ( l_sixsix_pre != 0 ) {
       buf[i++] = (unsigned char)0x66;
       l_place2++;
     }

     if (i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF )
     {
         if ( ( l_force_rex != 0 ) || l_gp8 || l_nx8 )
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
         if ( ( l_force_rex != 0 ) || l_gp8 || l_nx8 || l_ix8 )
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
       case LIBXSMM_X86_INSTR_SHLQ:
          if ( (i_immediate < 0) || (i_immediate > 127) )
          {
             fprintf(stderr,  "libxsmm_instruction_alu_imm is using an out-of-range immediate for salq.\n"
                              "because other immediates are signed but shlq is unsigned. So this code\n"
                              "should be changed if you want an immediate in this range.\n");
             exit(-1);
          }
          l_unsignedadj = 0x3e;
          l_third += 0x20;
          break;
       case LIBXSMM_X86_INSTR_SARQ:
          if ( (i_immediate < 0) || (i_immediate > 127) )
          {
             fprintf(stderr,  "libxsmm_instruction_alu_imm is using an out-of-range immediate for salq.\n"
                              "because other immediates are signed but sarq is unsigned. So this code\n"
                              "should be changed if you want an immediate in this range.\n");
             exit(-1);
          }
          l_unsignedadj = 0x3e;
          l_third += 0x38;
          break;
       case LIBXSMM_X86_INSTR_SHRQ:
          if ( (i_immediate < 0) || (i_immediate > 127) )
          {
             fprintf(stderr,  "libxsmm_instruction_alu_imm is using an out-of-range immediate for salq.\n"
                              "because other immediates are signed but shrq is unsigned. So this code\n"
                              "should be changed if you want an immediate in this range.\n");
             exit(-1);
          }
          l_unsignedadj = 0x3e;
          l_third += 0x28;
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
       case LIBXSMM_X86_INSTR_ANDQ:
          l_second += 0x20;
          l_third += 0x20;
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
       if ( i_gp_reg_number==0 && ((i_alu_instr==LIBXSMM_X86_INSTR_SUBQ) || (i_alu_instr==LIBXSMM_X86_INSTR_CMPQ) || (i_alu_instr==LIBXSMM_X86_INSTR_ADDQ) || (i_alu_instr==LIBXSMM_X86_INSTR_ANDQ)) )
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
       case LIBXSMM_X86_INSTR_ANDQ:
          l_second += 0x20;
          break;
       case LIBXSMM_X86_INSTR_CMOVA:
       case LIBXSMM_X86_INSTR_CMOVNBE:
          l_second += 0x0e;
          l_third += 0x03;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVAE:
       case LIBXSMM_X86_INSTR_CMOVNB:
       case LIBXSMM_X86_INSTR_CMOVNC:
          l_second += 0x0e;
          l_third -= 0x01;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVB:
       case LIBXSMM_X86_INSTR_CMOVC:
       case LIBXSMM_X86_INSTR_CMOVNAE:
          l_second += 0x0e;
          l_third -= 0x02;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVBE:
       case LIBXSMM_X86_INSTR_CMOVNA:
          l_second += 0x0e;
          l_third += 0x02;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVE:
       case LIBXSMM_X86_INSTR_CMOVZ:
          l_second += 0x0e;
          l_third += 0x00;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVG:
       case LIBXSMM_X86_INSTR_CMOVNLE:
          l_second += 0x0e;
          l_third += 0x0b;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVGE:
       case LIBXSMM_X86_INSTR_CMOVNL:
          l_second += 0x0e;
          l_third += 0x09;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVL:
       case LIBXSMM_X86_INSTR_CMOVNGE:
          l_second += 0x0e;
          l_third += 0x08;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVLE:
       case LIBXSMM_X86_INSTR_CMOVNG:
          l_second += 0x0e;
          l_third += 0x0a;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVNE:
       case LIBXSMM_X86_INSTR_CMOVNZ:
          l_second += 0x0e;
          l_third += 0x01;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVNO:
          l_second += 0x0e;
          l_third -= 0x03;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
        case LIBXSMM_X86_INSTR_CMOVNP:
        case LIBXSMM_X86_INSTR_CMOVPO:
          l_second += 0x0e;
          l_third += 0x07;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
        case LIBXSMM_X86_INSTR_CMOVNS:
          l_second += 0x0e;
          l_third += 0x05;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVO:
          l_second += 0x0e;
          l_third -= 0x04;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVP:
       case LIBXSMM_X86_INSTR_CMOVPE:
          l_second += 0x0e;
          l_third += 0x06;
          l_extra_byte = 1;
          l_reg1 = i_gp_reg_number_dest;
          l_reg0 = i_gp_reg_number_src;
          break;
       case LIBXSMM_X86_INSTR_CMOVS:
          l_second += 0x0e;
          l_third += 0x04;
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
  /* check if passed in a correct instruction */
  switch ( i_mask_instr ) {
    case LIBXSMM_X86_INSTR_KMOVB_GPR_LD:
    case LIBXSMM_X86_INSTR_KMOVW_GPR_LD:
    case LIBXSMM_X86_INSTR_KMOVD_GPR_LD:
    case LIBXSMM_X86_INSTR_KMOVQ_GPR_LD:
    case LIBXSMM_X86_INSTR_KMOVB_GPR_ST:
    case LIBXSMM_X86_INSTR_KMOVW_GPR_ST:
    case LIBXSMM_X86_INSTR_KMOVD_GPR_ST:
    case LIBXSMM_X86_INSTR_KMOVQ_GPR_ST:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_mask_move: unexpected instruction number: %u\n", i_mask_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    /* get L bit override */
    const libxsmm_x86_simd_name l_vname = ( (i_mask_instr & 0x300) == 0x300) ? LIBXSMM_X86_SIMD_NAME_YMM : LIBXSMM_X86_SIMD_NAME_XMM;
    unsigned int l_src;
    unsigned int l_dst;

    /* check if we need to flip operands */
    if ( ((i_mask_instr >> 24) & 0x08 ) == 0x08 ) {
      l_dst = i_gp_reg_number;
      l_src = i_mask_reg_number;
    } else {
      l_src = i_gp_reg_number;
      l_dst = i_mask_reg_number;
    }

    /* call vex encoder */
    libxsmm_x86_instruction_vex_compute_3reg( io_generated_code, i_mask_instr, l_vname,
                                              l_src, 0, l_dst );
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
         i_gp_reg_number == LIBXSMM_X86_GP_REG_R15) && (i_mask_instr != LIBXSMM_X86_INSTR_KMOVQ_GPR_LD) && (i_mask_instr != LIBXSMM_X86_INSTR_KMOVQ_GPR_ST) ) {
      l_prefix = 'd';
    }

    if ( ( i_mask_instr == LIBXSMM_X86_INSTR_KMOVB_GPR_ST ) ||
         ( i_mask_instr == LIBXSMM_X86_INSTR_KMOVW_GPR_ST ) ||
         ( i_mask_instr == LIBXSMM_X86_INSTR_KMOVD_GPR_ST ) ||
         ( i_mask_instr == LIBXSMM_X86_INSTR_KMOVQ_GPR_ST )    ) {
      if ( io_generated_code->code_type == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%k%u, %%%%%s%c\\n\\t\"\n", l_instr_name, i_mask_reg_number, l_gp_reg_name, l_prefix );
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%k%u, %%%s%c\n", l_instr_name, i_mask_reg_number, l_gp_reg_name, l_prefix );
      }
    } else {
      if ( io_generated_code->code_type == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%s%c, %%%%k%u\\n\\t\"\n", l_instr_name, l_gp_reg_name, l_prefix, i_mask_reg_number );
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%s%c, %%k%u\n", l_instr_name, l_gp_reg_name, l_prefix, i_mask_reg_number );
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_mask_move_mem( libxsmm_generated_code* io_generated_code,
                                            const unsigned int      i_mask_instr,
                                            const unsigned int      i_gp_reg_base,
                                            const unsigned int      i_gp_reg_idx,
                                            const unsigned int      i_scale,
                                            const int               i_displacement,
                                            const unsigned int      i_mask_reg_number ) {
  /* check if passed in a correct instruction */
  switch ( i_mask_instr ) {
    case LIBXSMM_X86_INSTR_KMOVB_LD:
    case LIBXSMM_X86_INSTR_KMOVW_LD:
    case LIBXSMM_X86_INSTR_KMOVD_LD:
    case LIBXSMM_X86_INSTR_KMOVQ_LD:
    case LIBXSMM_X86_INSTR_KMOVB_ST:
    case LIBXSMM_X86_INSTR_KMOVW_ST:
    case LIBXSMM_X86_INSTR_KMOVD_ST:
    case LIBXSMM_X86_INSTR_KMOVQ_ST:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_mask_move_mem: unexpected instruction number: %u\n", i_mask_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    /* get L bit override */
    const libxsmm_x86_simd_name l_vname = ( (i_mask_instr & 0x300) == 0x300) ? LIBXSMM_X86_SIMD_NAME_YMM : LIBXSMM_X86_SIMD_NAME_XMM;

    libxsmm_x86_instruction_vex_compute_2reg_mem( io_generated_code, i_mask_instr,
            i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, l_vname,
            0, i_mask_reg_number );
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_base[4];
    char l_gp_reg_idx[4];
    char l_instr_name[16];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base, 3 );
    libxsmm_get_x86_instr_name( i_mask_instr, l_instr_name, 15 );

    if ( ( i_mask_instr == LIBXSMM_X86_INSTR_KMOVB_ST ) ||
         ( i_mask_instr == LIBXSMM_X86_INSTR_KMOVW_ST ) ||
         ( i_mask_instr == LIBXSMM_X86_INSTR_KMOVD_ST ) ||
         ( i_mask_instr == LIBXSMM_X86_INSTR_KMOVQ_ST )    ) {
      if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF ) {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%k%u, %i(%%%%%s)\\n\\t\"\n", l_instr_name, i_mask_reg_number, i_displacement, l_gp_reg_base );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%k%u, %i(%%%s)\n", l_instr_name, i_mask_reg_number, i_displacement, l_gp_reg_base );
        }
      } else {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx, 3 );
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%k%u, %i(%%%%%s,%%%%%s,%u)\\n\\t\"\n", l_instr_name, i_mask_reg_number, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%k%u, %i(%%%s,%%%s,%u)\n", l_instr_name, i_mask_reg_number, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale );
        }
      }
    } else {
      if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF ) {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s), %%%%k%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, i_mask_reg_number );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%k%u\n", l_instr_name, i_displacement, l_gp_reg_base, i_mask_reg_number );
        }
      } else {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx, 3 );
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%%s,%u), %%%%k%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_mask_reg_number );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u), %%k%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_mask_reg_number );
        }
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_mask_compute_reg( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_mask_instr,
                                               const unsigned int      i_mask_reg_number_src_0,
                                               const unsigned int      i_mask_reg_number_src_1,
                                               const unsigned int      i_mask_reg_number_dest,
                                               const unsigned short    i_imm8 ) {
  /* check if passed in a correct instruction */
  switch ( i_mask_instr ) {
    case LIBXSMM_X86_INSTR_KADDB:
    case LIBXSMM_X86_INSTR_KADDW:
    case LIBXSMM_X86_INSTR_KADDD:
    case LIBXSMM_X86_INSTR_KADDQ:
    case LIBXSMM_X86_INSTR_KANDB:
    case LIBXSMM_X86_INSTR_KANDW:
    case LIBXSMM_X86_INSTR_KANDD:
    case LIBXSMM_X86_INSTR_KANDQ:
    case LIBXSMM_X86_INSTR_KANDNB:
    case LIBXSMM_X86_INSTR_KANDNW:
    case LIBXSMM_X86_INSTR_KANDND:
    case LIBXSMM_X86_INSTR_KANDNQ:
    case LIBXSMM_X86_INSTR_KNOTB:
    case LIBXSMM_X86_INSTR_KNOTW:
    case LIBXSMM_X86_INSTR_KNOTD:
    case LIBXSMM_X86_INSTR_KNOTQ:
    case LIBXSMM_X86_INSTR_KORB:
    case LIBXSMM_X86_INSTR_KORW:
    case LIBXSMM_X86_INSTR_KORD:
    case LIBXSMM_X86_INSTR_KORQ:
    case LIBXSMM_X86_INSTR_KORTESTB:
    case LIBXSMM_X86_INSTR_KORTESTW:
    case LIBXSMM_X86_INSTR_KORTESTD:
    case LIBXSMM_X86_INSTR_KORTESTQ:
    case LIBXSMM_X86_INSTR_KSHIFTLB:
    case LIBXSMM_X86_INSTR_KSHIFTLW:
    case LIBXSMM_X86_INSTR_KSHIFTLD:
    case LIBXSMM_X86_INSTR_KSHIFTLQ:
    case LIBXSMM_X86_INSTR_KSHIFTRB:
    case LIBXSMM_X86_INSTR_KSHIFTRW:
    case LIBXSMM_X86_INSTR_KSHIFTRD:
    case LIBXSMM_X86_INSTR_KSHIFTRQ:
    case LIBXSMM_X86_INSTR_KTESTB:
    case LIBXSMM_X86_INSTR_KTESTW:
    case LIBXSMM_X86_INSTR_KTESTD:
    case LIBXSMM_X86_INSTR_KTESTQ:
    case LIBXSMM_X86_INSTR_KUNPCKBW:
    case LIBXSMM_X86_INSTR_KUNPCKWD:
    case LIBXSMM_X86_INSTR_KUNPCKDQ:
    case LIBXSMM_X86_INSTR_KXNORB:
    case LIBXSMM_X86_INSTR_KXNORW:
    case LIBXSMM_X86_INSTR_KXNORD:
    case LIBXSMM_X86_INSTR_KXNORQ:
    case LIBXSMM_X86_INSTR_KXORB:
    case LIBXSMM_X86_INSTR_KXORW:
    case LIBXSMM_X86_INSTR_KXORD:
    case LIBXSMM_X86_INSTR_KXORQ:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_mask_compute_reg: unexpected instruction number: %u\n", i_mask_instr);
      exit(-1);
  }

  if ( io_generated_code->code_type > 1 ) {
    /* get L bit override */
    const libxsmm_x86_simd_name l_vname = ( (i_mask_instr & 0x300) == 0x300) ? LIBXSMM_X86_SIMD_NAME_YMM : LIBXSMM_X86_SIMD_NAME_XMM;
    unsigned int l_src1;

    /* check that we have an UNDEF for 2 src operands */
    if ( ((i_mask_instr >> 28) & 3) == 2 ) {
      if ( i_mask_reg_number_src_1 != LIBXSMM_X86_VEC_REG_UNDEF ) {
        fprintf(stderr, "libxsmm_x86_instruction_mask_compute_reg: In case of a 1 src operand instruction (%u), i_reg_number_src1 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_mask_instr);
        exit(-1);
      }
      l_src1 = 0;
    } else {
      l_src1 = i_mask_reg_number_src_1;
    }

    /* call vex encoder */
    libxsmm_x86_instruction_vex_compute_3reg( io_generated_code, i_mask_instr, l_vname,
                                              i_mask_reg_number_src_0, l_src1, i_mask_reg_number_dest );

    /* add imm if needed */
    if ( ((i_mask_instr >> 16) & 0x08) == 0x08 ) {
      if ( i_imm8 != LIBXSMM_X86_IMM_UNDEF ) {
        unsigned char* code = (unsigned char *) io_generated_code->generated_code;
        code[io_generated_code->code_size++] = (unsigned char)i_imm8;
      } else {
        fprintf(stderr, "libxsmm_x86_instruction_mask_compute_reg: imm8 required by instr, but LIBXSMM_X86_IMM_UNDEF was provided!\n");
        exit(-1);
      }
    }
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
void libxsmm_x86_instruction_tile_control( libxsmm_generated_code*    io_generated_code,
                                           const unsigned int         i_id,
                                           const unsigned int         i_instruction_set,
                                           const unsigned int         i_tcontrol_instr,
                                           const unsigned int         i_gp_reg_base,
                                           const int                  i_displacement,
                                           const libxsmm_tile_config* i_tile_config ) {
  char tile_config_imm[64];

  /* Can move these variables into the API if we choose: */
  /*const*/ unsigned int i_gp_reg_idx = LIBXSMM_X86_GP_REG_UNDEF;
  /*const*/ unsigned int i_scale = 1;

  /* @TODO: check instruction set */
  LIBXSMM_UNUSED( i_instruction_set );

  if ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_UNDEF) && (i_tile_config == NULL) && (i_tcontrol_instr != LIBXSMM_X86_INSTR_TILERELEASE) ) {
    fprintf(stderr, "invalid tile control!\n");
    exit(-1);
  }

  if (i_tcontrol_instr == LIBXSMM_X86_INSTR_LDTILECFG && i_tile_config != NULL) {
    unsigned int i;
    /* zeroing out imm structure as there are many reserved bytes */
    for ( i = 0; i < 64; i++ ) {
      tile_config_imm[i] = 0;
    }

    /* lets set tile_config_imm */
    tile_config_imm[0]                  = i_tile_config->palette_id;
    tile_config_imm[16]                 = (unsigned char)(0x00ff & i_tile_config->tile0rowsb);
    tile_config_imm[17]                 = (unsigned char)(0x00ff & (i_tile_config->tile0rowsb >> 8));
    tile_config_imm[48]                 = i_tile_config->tile0cols;
    tile_config_imm[18]                 = (unsigned char)(0x00ff & i_tile_config->tile1rowsb);
    tile_config_imm[19]                 = (unsigned char)(0x00ff & (i_tile_config->tile1rowsb >> 8));
    tile_config_imm[49]                 = i_tile_config->tile1cols;
    tile_config_imm[20]                 = (unsigned char)(0x00ff & i_tile_config->tile2rowsb);
    tile_config_imm[21]                 = (unsigned char)(0x00ff & (i_tile_config->tile2rowsb >> 8));
    tile_config_imm[50]                 = i_tile_config->tile2cols;
    tile_config_imm[22]                 = (unsigned char)(0x00ff & i_tile_config->tile3rowsb);
    tile_config_imm[23]                 = (unsigned char)(0x00ff & (i_tile_config->tile3rowsb >> 8));
    tile_config_imm[51]                 = i_tile_config->tile3cols;
    tile_config_imm[24]                 = (unsigned char)(0x00ff & i_tile_config->tile4rowsb);
    tile_config_imm[25]                 = (unsigned char)(0x00ff & (i_tile_config->tile4rowsb >> 8));
    tile_config_imm[52]                 = i_tile_config->tile4cols;
    tile_config_imm[26]                 = (unsigned char)(0x00ff & i_tile_config->tile5rowsb);
    tile_config_imm[27]                 = (unsigned char)(0x00ff & (i_tile_config->tile5rowsb >> 8));
    tile_config_imm[53]                 = i_tile_config->tile5cols;
    tile_config_imm[28]                 = (unsigned char)(0x00ff & i_tile_config->tile6rowsb);
    tile_config_imm[29]                 = (unsigned char)(0x00ff & (i_tile_config->tile6rowsb >> 8));
    tile_config_imm[54]                 = i_tile_config->tile6cols;
    tile_config_imm[30]                 = (unsigned char)(0x00ff & i_tile_config->tile7rowsb);
    tile_config_imm[31]                 = (unsigned char)(0x00ff & (i_tile_config->tile7rowsb >> 8));
    tile_config_imm[55]                 = i_tile_config->tile7cols;
  }

  /* @TODO add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* @Greg please add encodings here */
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    unsigned int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    unsigned int j;
    int l_regbas0 = i_gp_reg_base % 8;
    int l_gp8     = ((i_gp_reg_base > 7)&&(i_gp_reg_base<=15)?1:0);
    int l_third = 0;
    int l_fifth = 0;
    int l_place;
    int l_forced_offset = 0;

    if ( l_maxsize - i < 80 )
    {
       LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
       return;
    }

    switch ( i_tcontrol_instr ) {
      case LIBXSMM_X86_INSTR_LDTILECFG:
        /* here we need two cases:
         * a) gpr parameter is undefined, and i_tile_config != 0 -> use RIP as below in the assembly replacement
         * b) gpr parameter has some valid value and i_tile_config == 0 -> use gpr to address the tileconfig
         */
        break;
      case LIBXSMM_X86_INSTR_STTILECFG:
        /* here we alwyas it the gpr */
        l_third = 0x1;
        break;
      case LIBXSMM_X86_INSTR_TILERELEASE:
        l_fifth = 0xc0;
        l_gp8 = 0;
        l_regbas0 = 0;
        break;
      default:
        fprintf(stderr,"Unknown instruction in libxsmm_x86_instruction_tile_control. This is bad\n");
        break;
    }
#if 0
    if ( (i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF) && ((i_gp_reg_idx < LIBXSMM_X86_GP_REG_RAX) || (i_gp_reg_idx > LIBXSMM_X86_GP_REG_R15)) )
    {
       fprintf(stderr,"libxsmm_x86_instruction_tile_control is using a bogus i_gp_reg_idx\n");
       exit(-1);
    }
#endif
    if ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_UNDEF) && (i_tile_config != NULL) && (i_tcontrol_instr == LIBXSMM_X86_INSTR_LDTILECFG) )
    { /* Special case where we load from data segment */
       /* Jump past the next 64 bytes */
       buf[i++] = (unsigned char)(0xeb);
       buf[i++] = (unsigned char)(0x40);
       for ( j = i; j < i+64; j++ ) {
          buf[j] = (unsigned char)(tile_config_imm[j-i]);
       }
       i += 64;
       /* ldtilecfg .data1(%rip) where data1 is 64 bytes previous */
       buf[i++] = (unsigned char)(0xc4);
       buf[i++] = (unsigned char)(0xe2);
       buf[i++] = (unsigned char)(0x78);
       buf[i++] = (unsigned char)(0x49);
       buf[i++] = (unsigned char)(0x05);
       buf[i++] = (unsigned char)(0xb7);
       buf[i++] = (unsigned char)(0xff);
       buf[i++] = (unsigned char)(0xff);
       buf[i++] = (unsigned char)(0xff);
       io_generated_code->code_size = i;
    } else {
       if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF )
       {
          buf[i++] = (unsigned char)(0xc4);
          buf[i++] = (unsigned char)(0xe2 - l_gp8 * 0x20);
          buf[i++] = (unsigned char)(0x78 + l_third);
          buf[i++] = (unsigned char)(0x49);
          l_place = i - 1;
          buf[i++] = (unsigned char)(0x00 + l_regbas0 + l_fifth);
          if ( l_regbas0 == 4 ) buf[i++] = (unsigned char)(0x24);
       } else {
          int l_regidx  = i_gp_reg_idx  % 8;
          int l_ix8     = ((i_gp_reg_idx > 7)&&(i_gp_reg_idx<=15)?1:0);
          int l_sca=0;

          if (i_scale==2) l_sca=0x40;
          else if (i_scale==4) l_sca=0x80;
          else if (i_scale==8) l_sca=0xc0;

          buf[i++] = (unsigned char)(0xc4);
          buf[i++] = (unsigned char)(0xe2 - l_gp8 * 0x20 - l_ix8 * 0x40);
          buf[i++] = (unsigned char)(0x78 + l_third);
          buf[i++] = (unsigned char)(0x49);
          buf[i++] = (unsigned char)(0x04);
          l_place  = i - 1;
          buf[i++] = (unsigned char)(0x00 + l_sca + l_regbas0 + l_regidx*8);
       }

       if ( (l_regbas0 == 5) && (i_displacement==0) )
       {
           l_forced_offset = 1;
       }
       if ( i_tcontrol_instr != LIBXSMM_X86_INSTR_TILERELEASE )
       {
           /* All the other instructions have a memory offset */
           i += internal_x86_instructions_add_offset( l_place, i, i_displacement, l_forced_offset, 1, buf );
       }
       io_generated_code->code_size = i;
     }
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_base[4];
    unsigned int i;

    switch (i_tcontrol_instr) {
      case LIBXSMM_X86_INSTR_LDTILECFG:
      {
        if ( i_tile_config != NULL ) {
          if ( io_generated_code->code_type == 0 ) {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"jmp .continued_tconf_%u\\n\\t\"\n", i_id );
            libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \".data_tconf_%u:\\n\\t\"\n", i_id );
            libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
            for ( i = 0; i < 64; i += 4 ) {
              l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x\\n\\t\"\n",
                                                                                                          tile_config_imm[i  ], tile_config_imm[i+1],
                                                                                                          tile_config_imm[i+2], tile_config_imm[i+3]    );
              libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
            }
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \".continued_tconf_%u:\\n\\t\"\n", i_id );
            libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"ldtilecfg .data_tconf_%u(%%%%rip)\\n\\t\"\n", i_id );
            libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
          } else {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       jmp .continued_tconf_%u\n", i_id );
            libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       .data_tconf_%u:\n", i_id );
            libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
            for ( i = 0; i < 64; i += 4 ) {
              l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       .byte 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",
                                                                                                      tile_config_imm[i  ], tile_config_imm[i+1],
                                                                                                      tile_config_imm[i+2], tile_config_imm[i+3]    );
              libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
            }
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       .continued_tconf_%u:\n", i_id );
            libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       ldtilecfg .data_tconf_%u(%%rip)\n", i_id );
            libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
          }
        } else {
          if ( i_gp_reg_base != LIBXSMM_X86_GP_REG_UNDEF ) {
            libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base, 3 );
            if ( io_generated_code->code_type == 0 ) {
              l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"ldtilecfg %i(%%%%%s)\\n\\t\"\n",
                                                           i_displacement, l_gp_reg_base );
            } else {
              l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       ldtilecfg %i(%%%s)\n",
                                                           i_displacement, l_gp_reg_base );
            }
            libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
          } else {
            /* @TODO handle error */
          }
        }
        break;
      }
      case LIBXSMM_X86_INSTR_STTILECFG:
      {
        if ( i_gp_reg_base != LIBXSMM_X86_GP_REG_UNDEF ) {
          libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base, 3 );
          if ( io_generated_code->code_type == 0 ) {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"sttilecfg %i(%%%%%s)\\n\\t\"\n",
                                                         i_displacement, l_gp_reg_base );
          } else {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       sttilecfg %i(%%%s)\n",
                                                         i_displacement, l_gp_reg_base );
          }
          libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
        } else {
          /* @TODO handle error */
        }
        break;
      }
      case LIBXSMM_X86_INSTR_TILERELEASE:
      {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"tilerelease\\n\\t\"\n");
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       tilerelease\n");
        }
        break;
      }
      default:
      {
        /* this is an error */
        break;
      }
    }
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_tile_move( libxsmm_generated_code* io_generated_code,
                                        const unsigned int      i_instruction_set,
                                        const unsigned int      i_tmove_instr,
                                        const unsigned int      i_gp_reg_base,
                                        const unsigned int      i_gp_reg_idx,
                                        const unsigned int      i_scale,
                                        const int               i_displacement,
                                        const unsigned int      i_tile_reg_number ) {
  /* @TODO: check instruction set */
  LIBXSMM_UNUSED( i_instruction_set );

  /* check if passed in a correct instruction */
  switch ( i_tmove_instr ) {
    case LIBXSMM_X86_INSTR_TILELOADD:
    case LIBXSMM_X86_INSTR_TILELOADDT1:
    case LIBXSMM_X86_INSTR_TILESTORED:
    case LIBXSMM_X86_INSTR_TILEZERO:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_tile_move: unexpected instruction number: %u\n", i_tmove_instr);
      exit(-1);
  }

  if ( (io_generated_code->code_type > 1) &&
       (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) ) {
    /* check if we have enough code buffer space left */
    if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* invoke VEX encoder */
    if ( i_tmove_instr == LIBXSMM_X86_INSTR_TILEZERO ) {
      libxsmm_x86_instruction_vex_compute_3reg ( io_generated_code,
            i_tmove_instr, LIBXSMM_X86_SIMD_NAME_XMM, 0, 0, i_tile_reg_number );
    } else {
      if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF ) {
        libxsmm_x86_instruction_vex_compute_2reg_mem ( io_generated_code,
              i_tmove_instr, i_gp_reg_base, i_gp_reg_idx, i_scale,
              i_displacement, LIBXSMM_X86_SIMD_NAME_XMM, 0, i_tile_reg_number );
      } else {
        fprintf(stderr, "libxsmm_x86_instruction_tile_move: instruction %u requires SIB addressing\n", i_tmove_instr);
        exit(-1);
      }
    }
  } else if ( io_generated_code->code_type < 2 )  {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[24];
    char l_gp_reg_base[4];
    char l_gp_reg_idx[4];
    libxsmm_get_x86_instr_name( i_tmove_instr, l_instr_name, 23 );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_base, l_gp_reg_base, 3 );

    switch ( i_tmove_instr ) {
      case LIBXSMM_X86_INSTR_TILELOADD:
      case LIBXSMM_X86_INSTR_TILELOADDT1:
      {
        /* check that SIB addressing is set */
        if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF ) {
          libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx, 3 );
          if ( io_generated_code->code_type == 0 ) {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%%s,%u), %%%%tmm%u\\n\\t\"\n",
                                                         l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_tile_reg_number );
          } else {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u), %%tmm%u\n",
                                                         l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_tile_reg_number );
          }
          libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
        } else {
          /* @TODO handle error */
        }
        break;
      }
      case LIBXSMM_X86_INSTR_TILESTORED:
      {
        /* check that SIB addressing is set */
        if ( i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF ) {
          libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx, 3 );
          if ( io_generated_code->code_type == 0 ) {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%tmm%u, %i(%%%%%s,%%%%%s,%u)\\n\\t\"\n",
                                                         l_instr_name, i_tile_reg_number, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale );
          } else {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%tmm%u, %i(%%%s,%%%s,%u)\n",
                                                         l_instr_name, i_tile_reg_number, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale );
          }
          libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
        } else {
          /* @TODO handle error */
        }
        break;
      }
      case LIBXSMM_X86_INSTR_TILEZERO:
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%tmm%u\\n\\t\"\n",
                                                       l_instr_name, i_tile_reg_number );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%tmm%u\n",
                                                       l_instr_name, i_tile_reg_number );
        }
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      default:
        break;
    }
  } else {
    /* general encoder error */
    fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: GENERAL ERROR\n");
    exit(-1);
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_tile_compute( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_instruction_set,
                                           const unsigned int      i_tcompute_instr,
                                           const unsigned int      i_tile_src_reg_number_0,
                                           const unsigned int      i_tile_src_reg_number_1,
                                           const unsigned int      i_tile_dst_reg_number ) {
  /* @TODO: check instruction set */
  LIBXSMM_UNUSED( i_instruction_set );

  /* check if passed in a correct instruction */
  switch ( i_tcompute_instr ) {
    case LIBXSMM_X86_INSTR_TDPBSSD:
    case LIBXSMM_X86_INSTR_TDPBSUD:
    case LIBXSMM_X86_INSTR_TDPBUSD:
    case LIBXSMM_X86_INSTR_TDPBUUD:
    case LIBXSMM_X86_INSTR_TDPBF16PS:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_tile_compute: unexpected instruction number: %u\n", i_tcompute_instr);
      exit(-1);
  }

  if ( (io_generated_code->code_type > 1) &&
       (io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) ) {
    /* check if we have enough code buffer space left */
    if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* invoke VEX encoder */
    if ( ((i_tcompute_instr >> 28) & 0x3) == 3 ) {
      libxsmm_x86_instruction_vex_compute_3reg ( io_generated_code, i_tcompute_instr, LIBXSMM_X86_SIMD_NAME_XMM,
            i_tile_src_reg_number_1, i_tile_src_reg_number_0, i_tile_dst_reg_number );
    } else {
      fprintf(stderr, "libxsmm_x86_instruction_tile_compute: every insturction needs to have 3 operands\n");
      exit(-1);
    }
  } else if ( io_generated_code->code_type < 2 ) {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[24];
    libxsmm_get_x86_instr_name( i_tcompute_instr, l_instr_name, 23 );

    if ( io_generated_code->code_type == 0 ) {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%tmm%u, %%%%tmm%u, %%%%tmm%u\\n\\t\"\n",
                                                   l_instr_name, i_tile_src_reg_number_0, i_tile_src_reg_number_1, i_tile_dst_reg_number );
    } else {
      l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%tmm%u, %%tmm%u, %%tmm%u\n",
                                                   l_instr_name, i_tile_src_reg_number_0, i_tile_src_reg_number_1, i_tile_dst_reg_number );
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    /* general encoder error */
    fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: GENERAL ERROR\n");
    exit(-1);
  }
}

void libxsmm_x86_instruction_register_jump_back_label( libxsmm_generated_code*     io_generated_code,
                                                  libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  /* check if we still have label we can jump to */
  if ( io_loop_label_tracker->label_count == 512 ) {
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
  if ( i_label_no >= 512 ) {
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
  if ( (i_label_no < 512) == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  /* check if we still have label we can jump to */
  if ( io_jump_label_tracker->label_source[i_label_no].ref_count == 512-1 ) {
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
void libxsmm_x86_instruction_rdseed_load ( libxsmm_generated_code *io_generated_code,
                                           const unsigned int      i_gp_reg_number ) {
  if ( io_generated_code->code_type > 1 )
  {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    int i = io_generated_code->code_size;

    /* encode rdseed */
    unsigned char l_rex = (i_gp_reg_number < 8) ? 0x48 : 0x49;
    unsigned char l_pre = 0x0f;
    unsigned char l_op  = 0xc7;
    unsigned char l_modrm = (unsigned char)(0xf8 | (i_gp_reg_number & 0x7));
    buf[i++] = l_rex;
    buf[i++] = l_pre;
    buf[i++] = l_op;
    buf[i++] = l_modrm;

    /* jnc back 6 bytes -> rdseed was not ready jump back and retry */
    buf[i++] = 0x73;
    buf[i++] = 0xfa;
    /* reset CF, test al, al */
    buf[i++] = 0x84;
    buf[i++] = 0xc0;

    io_generated_code->code_size = i;
  } else {
    /* general encoder error */
    fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: GENERAL ERROR\n");
    exit(-1);
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
void libxsmm_x86_instruction_open_stream_amx( libxsmm_generated_code*   io_generated_code,
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
  } else if ( io_generated_code->code_type == 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

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
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream_amx( libxsmm_generated_code*   io_generated_code,
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

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream_mateltwise( libxsmm_generated_code*                   io_generated_code,
                                                     const unsigned int                        i_gp_struct_params,
                                                     int                                       skip_push ) {
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

    if (skip_push == 0) {
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
      /* adjust stack frame size */
      io_generated_code->sf_size += 40;
    }

    /* update code length */
    io_generated_code->code_size = l_code_size;
  } else if ( io_generated_code->code_type == 1 ) {
    /* @TODO this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    if (skip_push == 0) {

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

    }
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];

    /* loading struct params pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_struct_params, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "  __asm__ __volatile__(\"movq %%0, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream_mateltwise( libxsmm_generated_code*       io_generated_code,
                                                      int                           skip_pop) {
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is a very simple System V ABI 64 interface */
    unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (NULL == l_code_buffer || l_max_size < (l_code_size + 10)) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
    }

    if (skip_pop == 0) {
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
    }

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

    if (skip_pop == 0) {
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
    }

    /* @TODO: I don't know if this is the correct placement in the generation process */
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       retq\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    char l_new_code[1024];
    int l_max_code_length = 1023;
    int l_code_length = 0;

    if (io_generated_code->arch < LIBXSMM_X86_AVX512 ) {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(aptr), \"m\"(ldaptr), \"m\"(bptr), \"m\"(ldbptr), \"m\"(apfptr), \"m\"(bpfptr) : \"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n");
    } else {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(aptr), \"m\"(ldaptr), \"m\"(bptr), \"m\"(ldbptr), \"m\"(apfptr), \"m\"(bpfptr) : \"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream_matequation( libxsmm_generated_code*                  io_generated_code,
                                                     const unsigned int                        i_gp_struct_params ) {
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

    /* update code length */
    io_generated_code->code_size = l_code_size;
  } else if ( io_generated_code->code_type == 1 ) {
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_gp_reg_name[4];

    /* loading struct params pointer in assembly */
    libxsmm_get_x86_gp_reg_name( i_gp_struct_params, l_gp_reg_name, 3 );
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "  __asm__ __volatile__(\"movq %%0, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream_matequation( libxsmm_generated_code*       io_generated_code ) {
  if ( io_generated_code->code_type > 1 ) {
    /* @TODO this is a very simple System V ABI 64 interface */
    unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (NULL == l_code_buffer || l_max_size < (l_code_size + 10)) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
    }

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

    /* @TODO: I don't know if this is the correct placement in the generation process */
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       retq\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    char l_new_code[1024];
    int l_max_code_length = 1023;
    int l_code_length = 0;

    if (io_generated_code->arch < LIBXSMM_X86_AVX512 ) {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(aptr), \"m\"(ldaptr), \"m\"(bptr), \"m\"(ldbptr), \"m\"(apfptr), \"m\"(bpfptr) : \"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n");
    } else {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(aptr), \"m\"(ldaptr), \"m\"(bptr), \"m\"(ldbptr), \"m\"(apfptr), \"m\"(bpfptr) : \"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

