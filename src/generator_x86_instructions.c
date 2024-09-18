/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
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
    const unsigned char *l_cptr = (const unsigned char *)&i_offset;
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
     fprintf(stderr, "Bogus source location for internal jumping routine: %i\n", i_src_location);
     LIBXSMM_EXIT_ERROR(io_generated_code);
     return 0;
  }
  /* Make sure i_src_location is no bigger than the end of the code */
  if ( (unsigned int)i_src_location > io_generated_code->code_size )
  {
     fprintf(stderr, "How can the source of the jump itself be an instruction far beyond where we've jitted? Something is really strange here src=%i loc=%u\n", i_src_location, io_generated_code->code_size);
     LIBXSMM_EXIT_ERROR(io_generated_code);
     return 0;
  }

  if ( i_dest_location < 0 )
  {
     /* Must be a forward jump and we do not yet know it's dest location */
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

  /* Make sure we are not trying to jump to the same location as the original jump instruction */
  if ( i_src_location==i_dest_location || (i_src_location==i_dest_location+1) )
  {
     fprintf(stderr, "i_src_location=%i is physically too close to i_dest_location=%i\n",i_src_location,i_dest_location);
     LIBXSMM_EXIT_ERROR(io_generated_code);
     return 0;
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
unsigned int libxsmm_x86_instruction_vec_is_hybrid( const unsigned int i_instr ) {
  unsigned int l_return = 1;

  switch ( i_instr ) {
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
    case LIBXSMM_X86_INSTR_VPUNPCKLBW:
    case LIBXSMM_X86_INSTR_VPUNPCKHBW:
    case LIBXSMM_X86_INSTR_VPUNPCKLWD:
    case LIBXSMM_X86_INSTR_VPUNPCKHWD:
    case LIBXSMM_X86_INSTR_VPUNPCKLDQ:
    case LIBXSMM_X86_INSTR_VPUNPCKHDQ:
    case LIBXSMM_X86_INSTR_VPUNPCKLQDQ:
    case LIBXSMM_X86_INSTR_VPUNPCKHQDQ:
    case LIBXSMM_X86_INSTR_VEXTRACTF128:
    case LIBXSMM_X86_INSTR_VEXTRACTI128:
    case LIBXSMM_X86_INSTR_VPERMB:
    case LIBXSMM_X86_INSTR_VPERMW:
    case LIBXSMM_X86_INSTR_VPERMD:
    case LIBXSMM_X86_INSTR_VPERMQ_I:
    case LIBXSMM_X86_INSTR_VPERMPD:
    case LIBXSMM_X86_INSTR_VPERMPS:
    case LIBXSMM_X86_INSTR_VPERMPD_I:
    case LIBXSMM_X86_INSTR_VPERMILPS:
    case LIBXSMM_X86_INSTR_VPERMILPS_I:
    case LIBXSMM_X86_INSTR_VPERM2F128:
    case LIBXSMM_X86_INSTR_VPERM2I128:
    case LIBXSMM_X86_INSTR_VPERMILPD_VEX:
    case LIBXSMM_X86_INSTR_VPERMILPD_VEX_I:
    case LIBXSMM_X86_INSTR_VPERMILPD:
    case LIBXSMM_X86_INSTR_VPERMILPD_I:
    case LIBXSMM_X86_INSTR_VPERMT2B:
    case LIBXSMM_X86_INSTR_VPERMT2W:
    case LIBXSMM_X86_INSTR_VPERMT2D:
    case LIBXSMM_X86_INSTR_VPERMT2Q:
    case LIBXSMM_X86_INSTR_VPERMT2PS:
    case LIBXSMM_X86_INSTR_VPERMT2PD:
    case LIBXSMM_X86_INSTR_VPERMI2B:
    case LIBXSMM_X86_INSTR_VPERMI2W:
    case LIBXSMM_X86_INSTR_VPERMI2D:
    case LIBXSMM_X86_INSTR_VPERMI2Q:
    case LIBXSMM_X86_INSTR_VPERMI2PS:
    case LIBXSMM_X86_INSTR_VPERMI2PD:
    case LIBXSMM_X86_INSTR_VBLENDPD:
    case LIBXSMM_X86_INSTR_VBLENDPS:
    case LIBXSMM_X86_INSTR_VBLENDVPD:
    case LIBXSMM_X86_INSTR_VBLENDVPS:
    case LIBXSMM_X86_INSTR_VPBLENDD:
    case LIBXSMM_X86_INSTR_VPBLENDW:
    case LIBXSMM_X86_INSTR_VPBLENDVB:
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
    case LIBXSMM_X86_INSTR_VINSERTF32X4:
    case LIBXSMM_X86_INSTR_VINSERTF64X2:
    case LIBXSMM_X86_INSTR_VINSERTF32X8:
    case LIBXSMM_X86_INSTR_VINSERTF64X4:
    case LIBXSMM_X86_INSTR_VINSERTI32X4:
    case LIBXSMM_X86_INSTR_VINSERTI64X2:
    case LIBXSMM_X86_INSTR_VINSERTI32X8:
    case LIBXSMM_X86_INSTR_VINSERTI64X4:
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
    case LIBXSMM_X86_INSTR_VPSLLD_I:
    case LIBXSMM_X86_INSTR_VPSLLW_I:
    case LIBXSMM_X86_INSTR_VPSRAD_I:
    case LIBXSMM_X86_INSTR_VPSRAW_I:
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
    case LIBXSMM_X86_INSTR_VPXORQ:
    case LIBXSMM_X86_INSTR_VPORQ:
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
    case LIBXSMM_X86_INSTR_VPSUBW:
    case LIBXSMM_X86_INSTR_VPSUBB:
    case LIBXSMM_X86_INSTR_VPMAXSD:
    case LIBXSMM_X86_INSTR_VPMAXSW:
    case LIBXSMM_X86_INSTR_VPMINSD:
    case LIBXSMM_X86_INSTR_VPDPBUSD:
    case LIBXSMM_X86_INSTR_VPDPBUSDS:
    case LIBXSMM_X86_INSTR_VPDPWSSD:
    case LIBXSMM_X86_INSTR_VPDPWSSDS:
    case LIBXSMM_X86_INSTR_VDPBF16PS:
    case LIBXSMM_X86_INSTR_VCVTNEPS2BF16:
    case LIBXSMM_X86_INSTR_VCVTNE2PS2BF16:
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
    case LIBXSMM_X86_INSTR_VMOVDDUP:
    case LIBXSMM_X86_INSTR_VPMOVDB:
    case LIBXSMM_X86_INSTR_VPMOVWB:
    case LIBXSMM_X86_INSTR_VPMOVDW:
    case LIBXSMM_X86_INSTR_VPMOVSDB:
    case LIBXSMM_X86_INSTR_VPMOVSWB:
    case LIBXSMM_X86_INSTR_VPMOVSDW:
    case LIBXSMM_X86_INSTR_VPMOVUSDB:
    case LIBXSMM_X86_INSTR_VPMOVUSDW:
    case LIBXSMM_X86_INSTR_VPMOVUSWB:
    case LIBXSMM_X86_INSTR_VPMOVSXBW:
    case LIBXSMM_X86_INSTR_VPMOVSXBD:
    case LIBXSMM_X86_INSTR_VPMOVSXWD:
    case LIBXSMM_X86_INSTR_VPMOVZXWD:
    case LIBXSMM_X86_INSTR_VPMOVZXBW:
    case LIBXSMM_X86_INSTR_VPMOVZXBD:
    case LIBXSMM_X86_INSTR_VPACKSSWB:
    case LIBXSMM_X86_INSTR_VPACKSSDW:
    case LIBXSMM_X86_INSTR_VPACKUSWB:
    case LIBXSMM_X86_INSTR_VPACKUSDW:
    case LIBXSMM_X86_INSTR_VMOVD_LD:
    case LIBXSMM_X86_INSTR_VMOVQ_LD:
    case LIBXSMM_X86_INSTR_VMOVD_ST:
    case LIBXSMM_X86_INSTR_VMOVQ_ST:
    case LIBXSMM_X86_INSTR_MOVAPS_LD:
    case LIBXSMM_X86_INSTR_MOVUPS_LD:
    case LIBXSMM_X86_INSTR_MOVSS_LD:
    case LIBXSMM_X86_INSTR_MOVAPS_ST:
    case LIBXSMM_X86_INSTR_MOVUPS_ST:
    case LIBXSMM_X86_INSTR_MOVSS_ST:
    case LIBXSMM_X86_INSTR_ANDPS:
    case LIBXSMM_X86_INSTR_ANDNPS:
    case LIBXSMM_X86_INSTR_ORPS:
    case LIBXSMM_X86_INSTR_XORPS:
    case LIBXSMM_X86_INSTR_MULPS:
    case LIBXSMM_X86_INSTR_ADDPS:
    case LIBXSMM_X86_INSTR_SUBPS:
    case LIBXSMM_X86_INSTR_DIVPS:
    case LIBXSMM_X86_INSTR_RCPPS:
    case LIBXSMM_X86_INSTR_SQRTPS:
    case LIBXSMM_X86_INSTR_MAXPS:
    case LIBXSMM_X86_INSTR_MINPS:
    case LIBXSMM_X86_INSTR_RSQRTPS:
    case LIBXSMM_X86_INSTR_CMPPS:
    case LIBXSMM_X86_INSTR_SHUFPS:
    case LIBXSMM_X86_INSTR_UNPCKHPS:
    case LIBXSMM_X86_INSTR_UNPCKLPS:
    case LIBXSMM_X86_INSTR_MULSS:
    case LIBXSMM_X86_INSTR_ADDSS:
    case LIBXSMM_X86_INSTR_SUBSS:
    case LIBXSMM_X86_INSTR_DIVSS:
    case LIBXSMM_X86_INSTR_RCPSS:
    case LIBXSMM_X86_INSTR_SQRTSS:
    case LIBXSMM_X86_INSTR_MAXSS:
    case LIBXSMM_X86_INSTR_MINSS:
    case LIBXSMM_X86_INSTR_RSQRTSS:
    case LIBXSMM_X86_INSTR_CMPSS:
    case LIBXSMM_X86_INSTR_COMISS:
    case LIBXSMM_X86_INSTR_UCOMISS:
    case LIBXSMM_X86_INSTR_MOVAPD_LD:
    case LIBXSMM_X86_INSTR_MOVUPD_LD:
    case LIBXSMM_X86_INSTR_MOVSD_LD:
    case LIBXSMM_X86_INSTR_MOVAPD_ST:
    case LIBXSMM_X86_INSTR_MOVUPD_ST:
    case LIBXSMM_X86_INSTR_MOVSD_ST:
    case LIBXSMM_X86_INSTR_ANDPD:
    case LIBXSMM_X86_INSTR_ANDNPD:
    case LIBXSMM_X86_INSTR_ORPD:
    case LIBXSMM_X86_INSTR_XORPD:
    case LIBXSMM_X86_INSTR_MULPD:
    case LIBXSMM_X86_INSTR_ADDPD:
    case LIBXSMM_X86_INSTR_SUBPD:
    case LIBXSMM_X86_INSTR_DIVPD:
    case LIBXSMM_X86_INSTR_RCPPD:
    case LIBXSMM_X86_INSTR_SQRTPD:
    case LIBXSMM_X86_INSTR_MAXPD:
    case LIBXSMM_X86_INSTR_MINPD:
    case LIBXSMM_X86_INSTR_RSQRTPD:
    case LIBXSMM_X86_INSTR_CMPPD:
    case LIBXSMM_X86_INSTR_SHUFPD:
    case LIBXSMM_X86_INSTR_UNPCKHPD:
    case LIBXSMM_X86_INSTR_UNPCKLPD:
    case LIBXSMM_X86_INSTR_MULSD:
    case LIBXSMM_X86_INSTR_ADDSD:
    case LIBXSMM_X86_INSTR_SUBSD:
    case LIBXSMM_X86_INSTR_DIVSD:
    case LIBXSMM_X86_INSTR_RCPSD:
    case LIBXSMM_X86_INSTR_SQRTSD:
    case LIBXSMM_X86_INSTR_MAXSD:
    case LIBXSMM_X86_INSTR_MINSD:
    case LIBXSMM_X86_INSTR_RSQRTSD:
    case LIBXSMM_X86_INSTR_CMPSD:
    case LIBXSMM_X86_INSTR_COMISD:
    case LIBXSMM_X86_INSTR_UCOMISD:
    case LIBXSMM_X86_INSTR_MOVD_SSE_LD:
    case LIBXSMM_X86_INSTR_MOVD_SSE_ST:
    case LIBXSMM_X86_INSTR_MOVQ_SSE_LD:
    case LIBXSMM_X86_INSTR_MOVQ_SSE_ST:
    case LIBXSMM_X86_INSTR_MOVDQA_LD:
    case LIBXSMM_X86_INSTR_MOVDQA_ST:
    case LIBXSMM_X86_INSTR_MOVDQU_LD:
    case LIBXSMM_X86_INSTR_MOVDQU_ST:
    case LIBXSMM_X86_INSTR_MOVDDUP:
    case LIBXSMM_X86_INSTR_MOVSHDUP:
    case LIBXSMM_X86_INSTR_MOVSLDUP:
    case LIBXSMM_X86_INSTR_PAND:
    case LIBXSMM_X86_INSTR_PANDN:
    case LIBXSMM_X86_INSTR_POR:
    case LIBXSMM_X86_INSTR_PXOR:
    case LIBXSMM_X86_INSTR_PACKSSWB:
    case LIBXSMM_X86_INSTR_PACKSSDW:
    case LIBXSMM_X86_INSTR_PACKUSWB:
    case LIBXSMM_X86_INSTR_PADDB:
    case LIBXSMM_X86_INSTR_PADDW:
    case LIBXSMM_X86_INSTR_PADDD:
    case LIBXSMM_X86_INSTR_PADDQ:
    case LIBXSMM_X86_INSTR_PADDSB:
    case LIBXSMM_X86_INSTR_PADDSW:
    case LIBXSMM_X86_INSTR_PADDUSB:
    case LIBXSMM_X86_INSTR_PADDUSW:
    case LIBXSMM_X86_INSTR_PAVGB:
    case LIBXSMM_X86_INSTR_PAVGW:
    case LIBXSMM_X86_INSTR_PCMPEQB:
    case LIBXSMM_X86_INSTR_PCMPEQW:
    case LIBXSMM_X86_INSTR_PCMPEQD:
    case LIBXSMM_X86_INSTR_PCMPGTB:
    case LIBXSMM_X86_INSTR_PCMPGTW:
    case LIBXSMM_X86_INSTR_PCMPGTD:
    case LIBXSMM_X86_INSTR_PEXTRW:
    case LIBXSMM_X86_INSTR_PINSRW:
    case LIBXSMM_X86_INSTR_PMADDWD:
    case LIBXSMM_X86_INSTR_PMAXSW:
    case LIBXSMM_X86_INSTR_PMAXUB:
    case LIBXSMM_X86_INSTR_PMINSW:
    case LIBXSMM_X86_INSTR_PMINUB:
    case LIBXSMM_X86_INSTR_PMULHUW:
    case LIBXSMM_X86_INSTR_PMULHW:
    case LIBXSMM_X86_INSTR_PMULLW:
    case LIBXSMM_X86_INSTR_PMULUDQ:
    case LIBXSMM_X86_INSTR_PSADBW:
    case LIBXSMM_X86_INSTR_PSHUFD:
    case LIBXSMM_X86_INSTR_PSHUFHW:
    case LIBXSMM_X86_INSTR_PSHUFLW:
    case LIBXSMM_X86_INSTR_PSLLW:
    case LIBXSMM_X86_INSTR_PSLLD:
    case LIBXSMM_X86_INSTR_PSLLQ:
    case LIBXSMM_X86_INSTR_PSRAW:
    case LIBXSMM_X86_INSTR_PSRAD:
    case LIBXSMM_X86_INSTR_PSRLW:
    case LIBXSMM_X86_INSTR_PSRLD:
    case LIBXSMM_X86_INSTR_PSRLQ:
    case LIBXSMM_X86_INSTR_PSUBB:
    case LIBXSMM_X86_INSTR_PSUBW:
    case LIBXSMM_X86_INSTR_PSUBD:
    case LIBXSMM_X86_INSTR_PSUBQ:
    case LIBXSMM_X86_INSTR_PSUBSB:
    case LIBXSMM_X86_INSTR_PSUBSW:
    case LIBXSMM_X86_INSTR_PSUBUSB:
    case LIBXSMM_X86_INSTR_PSUBUSW:
    case LIBXSMM_X86_INSTR_PUNPCKHBW:
    case LIBXSMM_X86_INSTR_PUNPCKHWD:
    case LIBXSMM_X86_INSTR_PUNPCKHDQ:
    case LIBXSMM_X86_INSTR_PUNPCKHQDQ:
    case LIBXSMM_X86_INSTR_PUNPCKLBW:
    case LIBXSMM_X86_INSTR_PUNPCKLWD:
    case LIBXSMM_X86_INSTR_PUNPCKLDQ:
    case LIBXSMM_X86_INSTR_PUNPCKLQDQ:
    case LIBXSMM_X86_INSTR_CVTDQ2PD:
    case LIBXSMM_X86_INSTR_CVTDQ2PS:
    case LIBXSMM_X86_INSTR_CVTPD2DQ:
    case LIBXSMM_X86_INSTR_CVTPD2PS:
    case LIBXSMM_X86_INSTR_CVTPS2DQ:
    case LIBXSMM_X86_INSTR_CVTPS2PD:
    case LIBXSMM_X86_INSTR_CVTSD2SS:
    case LIBXSMM_X86_INSTR_CVTSS2SD:
    case LIBXSMM_X86_INSTR_CVTTPD2DQ:
    case LIBXSMM_X86_INSTR_CVTTPS2DQ:
    case LIBXSMM_X86_INSTR_ADDSUBPD:
    case LIBXSMM_X86_INSTR_ADDSUBPS:
    case LIBXSMM_X86_INSTR_HADDPD:
    case LIBXSMM_X86_INSTR_HADDPS:
    case LIBXSMM_X86_INSTR_HSUBPD:
    case LIBXSMM_X86_INSTR_HSUBPS:
    case LIBXSMM_X86_INSTR_PABSB:
    case LIBXSMM_X86_INSTR_PABSW:
    case LIBXSMM_X86_INSTR_PABSD:
    case LIBXSMM_X86_INSTR_PALIGNR:
    case LIBXSMM_X86_INSTR_PHADDW:
    case LIBXSMM_X86_INSTR_PHADDD:
    case LIBXSMM_X86_INSTR_PHADDSW:
    case LIBXSMM_X86_INSTR_PHSUBW:
    case LIBXSMM_X86_INSTR_PHSUBD:
    case LIBXSMM_X86_INSTR_PHSUBSW:
    case LIBXSMM_X86_INSTR_PMADDUBSW:
    case LIBXSMM_X86_INSTR_PMULHRSW:
    case LIBXSMM_X86_INSTR_PSHUFB:
    case LIBXSMM_X86_INSTR_PSIGNB:
    case LIBXSMM_X86_INSTR_PSIGNW:
    case LIBXSMM_X86_INSTR_PSIGND:
    case LIBXSMM_X86_INSTR_BLENDPD:
    case LIBXSMM_X86_INSTR_BLENDPS:
    case LIBXSMM_X86_INSTR_BLENDVPD:
    case LIBXSMM_X86_INSTR_BLENDVPS:
    case LIBXSMM_X86_INSTR_DPPD:
    case LIBXSMM_X86_INSTR_DPPS:
    case LIBXSMM_X86_INSTR_EXTRACTPS:
    case LIBXSMM_X86_INSTR_INSERTPS:
    case LIBXSMM_X86_INSTR_ROUNDPD:
    case LIBXSMM_X86_INSTR_ROUNDPS:
    case LIBXSMM_X86_INSTR_ROUNDSD:
    case LIBXSMM_X86_INSTR_ROUNDSS:
    case LIBXSMM_X86_INSTR_PBLENDW:
    case LIBXSMM_X86_INSTR_PBLENDVB:
    case LIBXSMM_X86_INSTR_PCMPEQQ:
    case LIBXSMM_X86_INSTR_PMOVSXBW:
    case LIBXSMM_X86_INSTR_PMOVSXBD:
    case LIBXSMM_X86_INSTR_PMOVSXBQ:
    case LIBXSMM_X86_INSTR_PMOVSXWD:
    case LIBXSMM_X86_INSTR_PMOVSXWQ:
    case LIBXSMM_X86_INSTR_PMOVSXDQ:
    case LIBXSMM_X86_INSTR_PMOVZXBW:
    case LIBXSMM_X86_INSTR_PMOVZXBD:
    case LIBXSMM_X86_INSTR_PMOVZXBQ:
    case LIBXSMM_X86_INSTR_PMOVZXWD:
    case LIBXSMM_X86_INSTR_PMOVZXWQ:
    case LIBXSMM_X86_INSTR_PMOVZXDQ:
    case LIBXSMM_X86_INSTR_PEXTRB:
    case LIBXSMM_X86_INSTR_PEXTRD:
    case LIBXSMM_X86_INSTR_PEXTRQ:
    case LIBXSMM_X86_INSTR_PHMINPOSUW:
    case LIBXSMM_X86_INSTR_PINSRB:
    case LIBXSMM_X86_INSTR_PINSRD:
    case LIBXSMM_X86_INSTR_PINSRQ:
    case LIBXSMM_X86_INSTR_PMAXSB:
    case LIBXSMM_X86_INSTR_PMAXSD:
    case LIBXSMM_X86_INSTR_PMAXUW:
    case LIBXSMM_X86_INSTR_PMAXUD:
    case LIBXSMM_X86_INSTR_PMINSB:
    case LIBXSMM_X86_INSTR_PMINSD:
    case LIBXSMM_X86_INSTR_PMINUW:
    case LIBXSMM_X86_INSTR_PMINUD:
    case LIBXSMM_X86_INSTR_MPSADBW:
    case LIBXSMM_X86_INSTR_PMULDQ:
    case LIBXSMM_X86_INSTR_PMULLD:
    case LIBXSMM_X86_INSTR_PACKUSDW:
    case LIBXSMM_X86_INSTR_PTEST:
    case LIBXSMM_X86_INSTR_PCMPGTQ:
    case LIBXSMM_X86_INSTR_VBROADCASTSD:
    case LIBXSMM_X86_INSTR_VBROADCASTSS:
    case LIBXSMM_X86_INSTR_VBROADCASTSD_VEX:
    case LIBXSMM_X86_INSTR_VBROADCASTI32X2:
    case LIBXSMM_X86_INSTR_VPBROADCASTD:
    case LIBXSMM_X86_INSTR_VPBROADCASTQ:
    case LIBXSMM_X86_INSTR_VPBROADCASTQ_VEX:
    case LIBXSMM_X86_INSTR_VPBROADCASTB:
    case LIBXSMM_X86_INSTR_VPBROADCASTW:
    case LIBXSMM_X86_INSTR_VADDPH:
    case LIBXSMM_X86_INSTR_VADDSH:
    case LIBXSMM_X86_INSTR_VCMPPH:
    case LIBXSMM_X86_INSTR_VCMPSH:
    case LIBXSMM_X86_INSTR_VDIVPH:
    case LIBXSMM_X86_INSTR_VDIVSH:
    case LIBXSMM_X86_INSTR_VFCMADDCPH:
    case LIBXSMM_X86_INSTR_VFMADDCPH:
    case LIBXSMM_X86_INSTR_VFCMADDCSH:
    case LIBXSMM_X86_INSTR_VFMADDCSH:
    case LIBXSMM_X86_INSTR_VFCMULCPH:
    case LIBXSMM_X86_INSTR_VFMULCPH:
    case LIBXSMM_X86_INSTR_VFCMULCSH:
    case LIBXSMM_X86_INSTR_VFMULCSH:
    case LIBXSMM_X86_INSTR_VFMADDSUB132PH:
    case LIBXSMM_X86_INSTR_VFMADDSUB213PH:
    case LIBXSMM_X86_INSTR_VFMADDSUB231PH:
    case LIBXSMM_X86_INSTR_VFMSUBADD132PH:
    case LIBXSMM_X86_INSTR_VFMSUBADD213PH:
    case LIBXSMM_X86_INSTR_VFMSUBADD231PH:
    case LIBXSMM_X86_INSTR_VFMADD132PH:
    case LIBXSMM_X86_INSTR_VFMADD213PH:
    case LIBXSMM_X86_INSTR_VFMADD231PH:
    case LIBXSMM_X86_INSTR_VFNMADD132PH:
    case LIBXSMM_X86_INSTR_VFNMADD213PH:
    case LIBXSMM_X86_INSTR_VFNMADD231PH:
    case LIBXSMM_X86_INSTR_VFMADD132SH:
    case LIBXSMM_X86_INSTR_VFMADD213SH:
    case LIBXSMM_X86_INSTR_VFMADD231SH:
    case LIBXSMM_X86_INSTR_VFNMADD132SH:
    case LIBXSMM_X86_INSTR_VFNMADD213SH:
    case LIBXSMM_X86_INSTR_VFNMADD231SH:
    case LIBXSMM_X86_INSTR_VFMSUB132PH:
    case LIBXSMM_X86_INSTR_VFMSUB213PH:
    case LIBXSMM_X86_INSTR_VFMSUB231PH:
    case LIBXSMM_X86_INSTR_VFNMSUB132PH:
    case LIBXSMM_X86_INSTR_VFNMSUB213PH:
    case LIBXSMM_X86_INSTR_VFNMSUB231PH:
    case LIBXSMM_X86_INSTR_VFMSUB132SH:
    case LIBXSMM_X86_INSTR_VFMSUB213SH:
    case LIBXSMM_X86_INSTR_VFMSUB231SH:
    case LIBXSMM_X86_INSTR_VFNMSUB132SH:
    case LIBXSMM_X86_INSTR_VFNMSUB213SH:
    case LIBXSMM_X86_INSTR_VFNMSUB231SH:
    case LIBXSMM_X86_INSTR_VPCLASSPH:
    case LIBXSMM_X86_INSTR_VPCLASSSH:
    case LIBXSMM_X86_INSTR_VGETEXPPH:
    case LIBXSMM_X86_INSTR_VGETEXPSH:
    case LIBXSMM_X86_INSTR_VGETMANTPH:
    case LIBXSMM_X86_INSTR_VGETMANTSH:
    case LIBXSMM_X86_INSTR_VMAXPH:
    case LIBXSMM_X86_INSTR_VMAXSH:
    case LIBXSMM_X86_INSTR_VMINPH:
    case LIBXSMM_X86_INSTR_VMINSH:
    case LIBXSMM_X86_INSTR_VMOVW_LD:
    case LIBXSMM_X86_INSTR_VMOVW_ST:
    case LIBXSMM_X86_INSTR_VMULPH:
    case LIBXSMM_X86_INSTR_VMULSH:
    case LIBXSMM_X86_INSTR_VRCPPH:
    case LIBXSMM_X86_INSTR_VRCPSH:
    case LIBXSMM_X86_INSTR_VREDUCEPH:
    case LIBXSMM_X86_INSTR_VREDUCESH:
    case LIBXSMM_X86_INSTR_VRNDSCALEPH:
    case LIBXSMM_X86_INSTR_VRNDSCALESH:
    case LIBXSMM_X86_INSTR_VRSQRTPH:
    case LIBXSMM_X86_INSTR_VRSQRTSH:
    case LIBXSMM_X86_INSTR_VSCALEFPH:
    case LIBXSMM_X86_INSTR_VSCALEFSH:
    case LIBXSMM_X86_INSTR_VSQRTPH:
    case LIBXSMM_X86_INSTR_VSQRTSH:
    case LIBXSMM_X86_INSTR_VSUBPH:
    case LIBXSMM_X86_INSTR_VSUBSH:
    case LIBXSMM_X86_INSTR_VCVTW2PH:
    case LIBXSMM_X86_INSTR_VPDPBSUD:
    case LIBXSMM_X86_INSTR_VPDPBSUDS:
    case LIBXSMM_X86_INSTR_VPDPBSSD:
    case LIBXSMM_X86_INSTR_VPDPBSSDS:
    case LIBXSMM_X86_INSTR_VPDPBUUD:
    case LIBXSMM_X86_INSTR_VPDPBUUDS:
      break;
    default:
      l_return = 0;
  }

  return l_return;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_x86_instruction_vec_is_regmemonly( const unsigned int i_instr ) {
  unsigned int l_return = 1;

  switch ( i_instr ) {
    case LIBXSMM_X86_INSTR_VBROADCASTI128:
    case LIBXSMM_X86_INSTR_VBROADCASTI32X4:
    case LIBXSMM_X86_INSTR_VBROADCASTI64X2:
    case LIBXSMM_X86_INSTR_VBROADCASTI32X8:
    case LIBXSMM_X86_INSTR_VBROADCASTI64X4:
    case LIBXSMM_X86_INSTR_VMOVNTPD:
    case LIBXSMM_X86_INSTR_VMOVNTPS:
    case LIBXSMM_X86_INSTR_VMOVNTDQ:
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
    case LIBXSMM_X86_INSTR_MOVLPS:
    case LIBXSMM_X86_INSTR_MOVHPS:
    case LIBXSMM_X86_INSTR_MOVNTPS:
    case LIBXSMM_X86_INSTR_MOVLPD:
    case LIBXSMM_X86_INSTR_MOVHPD:
    case LIBXSMM_X86_INSTR_MOVNTPD:
    case LIBXSMM_X86_INSTR_MOVNTDQ:
    case LIBXSMM_X86_INSTR_MOVNTDQA:
    case LIBXSMM_X86_INSTR_LDDQU:
    case LIBXSMM_X86_INSTR_VMOVSH_LD_MEM:
    case LIBXSMM_X86_INSTR_VMOVSH_ST_MEM:
    case LIBXSMM_X86_INSTR_VBCSTNEBF162PS:
    case LIBXSMM_X86_INSTR_VBCSTNESH2PS:
    case LIBXSMM_X86_INSTR_VCVTNEEBF162PS:
    case LIBXSMM_X86_INSTR_VCVTNEEPH2PS:
    case LIBXSMM_X86_INSTR_VCVTNEOBF162PS:
    case LIBXSMM_X86_INSTR_VCVTNEOPH2PS:
    case LIBXSMM_X86_INSTR_VPEXTRB:
    case LIBXSMM_X86_INSTR_VPEXTRD:
    case LIBXSMM_X86_INSTR_VPEXTRQ:
    case LIBXSMM_X86_INSTR_VPINSRB:
    case LIBXSMM_X86_INSTR_VPINSRD:
    case LIBXSMM_X86_INSTR_VPINSRQ:
    case LIBXSMM_X86_INSTR_VPERMQ:
      break;
    default:
      l_return = 0;
  }

  return l_return;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_x86_instruction_vec_is_regonly( const unsigned int i_instr ) {
  unsigned int l_return = 1;

  switch ( i_instr ) {
    case LIBXSMM_X86_INSTR_VMOVMSKPD:
    case LIBXSMM_X86_INSTR_VMOVMSKPS:
    case LIBXSMM_X86_INSTR_VPMOVMSKB:
    case LIBXSMM_X86_INSTR_VPBROADCASTB_GPR:
    case LIBXSMM_X86_INSTR_VPBROADCASTW_GPR:
    case LIBXSMM_X86_INSTR_VPBROADCASTD_GPR:
    case LIBXSMM_X86_INSTR_VPBROADCASTQ_GPR:
    case LIBXSMM_X86_INSTR_MOVMSKPS:
    case LIBXSMM_X86_INSTR_MOVMSKPD:
    case LIBXSMM_X86_INSTR_PMOVMSKB:
    case LIBXSMM_X86_INSTR_PSLLW_I:
    case LIBXSMM_X86_INSTR_PSLLD_I:
    case LIBXSMM_X86_INSTR_PSLLQ_I:
    case LIBXSMM_X86_INSTR_PSLLDQ_I:
    case LIBXSMM_X86_INSTR_PSRAW_I:
    case LIBXSMM_X86_INSTR_PSRAD_I:
    case LIBXSMM_X86_INSTR_PSRLDQ_I:
    case LIBXSMM_X86_INSTR_PSRLW_I:
    case LIBXSMM_X86_INSTR_PSRLD_I:
    case LIBXSMM_X86_INSTR_PSRLQ_I:
    case LIBXSMM_X86_INSTR_VMOVSH_LD_3REG:
    case LIBXSMM_X86_INSTR_VMOVSH_ST_3REG:
      break;
    default:
      l_return = 0;
  }

  return l_return;
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_rex_compute_1reg_mem( libxsmm_generated_code*     io_generated_code,
                                                   const unsigned int          i_instr,
                                                   const unsigned int          i_gp_reg_base,
                                                   const unsigned int          i_gp_reg_idx,
                                                   const unsigned int          i_scale,
                                                   const int                   i_displacement,
                                                   const unsigned int          i_reg_number_reg )
{
  unsigned int code_head = io_generated_code->code_size;
  unsigned char* code    = (unsigned char *)io_generated_code->generated_code;
  /* prefix and op-code tables */
  unsigned char tbl_prefix[4] = {0x00, 0x66, 0xf3, 0xf2};
  unsigned char tbl_opext[2] = {0x38, 0x3a};
  unsigned int prefix_idx = ((i_instr & 0x00070000) >> 16) - 4;
  unsigned int opext = (i_instr & 0x00003000) >> 12;
  unsigned int opext_idx = (i_instr & 0x00001000) >> 12;
  /* table for modrm mod settings */
  unsigned char tbl_scale[9] = { 0x00, 0x00, 0x40, 0x40, 0x80, 0x80, 0x80, 0x80, 0xc0 };
  /* storing pointer to modrm byte */
  unsigned int modrm = 0;
  /* control variable if we need to encode in SIB mode */
  unsigned char l_have_sib = 0;
  /* when having RBP/R13 as base register, we need a SIB byte, even without idx GPR */
  unsigned char l_forced_zdisp8 = 0;
  /* we need a local non-const i_gp_reg_idx copy */
  unsigned int l_gp_reg_idx;
  /* we need a local non-const i_scale copy */
  unsigned int l_scale;

  /* check if we have enough code buffer space left */
  if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
    return;
  }

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

  /* 1 B) determining if a force zero displacement is needed */
  if ( ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_RBP) || (i_gp_reg_base == LIBXSMM_X86_GP_REG_R13) ) && (i_displacement == 0) ) {
    l_forced_zdisp8 = 1;
  } else {
    l_forced_zdisp8 = 0;
  }

  /* encoding */
  /* A): writing prefixes */
  /* operand size overwrite prefix */
  if ( (i_instr & 0x0000c000) == 0x00004000 ) {
    code[code_head++] = 0x66;
  }
  /* instruction prefix */
  if ( prefix_idx != 0 ) {
    code[code_head++] = tbl_prefix[prefix_idx];
  }
  /* REX prefix */
  if ( (i_reg_number_reg > 7) || (i_gp_reg_base > 7) || ( (l_gp_reg_idx > 7) && (l_have_sib == 1) ) || ( (i_instr & 0x02000000) == 0x02000000 ) ) {
    /* R */
    code[code_head  ]  = (unsigned char)(( i_reg_number_reg > 7 ) ? 0x04 : 0x00);
    /* B */
    code[code_head  ] |= (unsigned char)(( i_gp_reg_base > 7 ) ? 0x01 : 0x00);
    /* W */
    code[code_head  ] |= (unsigned char)(( (i_instr & 0x00800000) == 0x00800000 ) ? 0x08 : 0x00);
    /* when have SIB, set Z */
    if ( l_have_sib == 1 ) {
      code[code_head  ] |= (unsigned char)(( l_gp_reg_idx > 7 ) ? 0x02 : 0x00);
    }
    /* start of REX prefix */
    code[code_head++] |= 0x40;
  }

  /* B): opcode extensions */
  if ( opext > 0 ) {
    code[code_head++] = 0x0f;
    if ( opext > 1 ) {
      code[code_head++] = tbl_opext[opext_idx];
    }
  }

  /* C) writing lowest op code byte */
  code[code_head++] = (unsigned char) (i_instr & 0x000000ff);

  /* D) setting modrm, we are in reg-only addressing mode */
  modrm = code_head;
  code[code_head  ]  = (unsigned char)(((unsigned char)(i_reg_number_reg << 3)) & 0x38);
  if ( l_have_sib == 1 ) {
    code[code_head++] |= (unsigned char)0x04; /* set SIB mode*/
    /* set SIB */
    code[code_head  ]  = tbl_scale[l_scale];
    code[code_head  ] |= (unsigned char)(((unsigned char)(l_gp_reg_idx << 3)) & 0x38);
    code[code_head++] |= (unsigned char)(((unsigned char) i_gp_reg_base  )    & 0x07);
  } else {
    code[code_head++] |= (unsigned char)(((unsigned char) i_gp_reg_base)      & 0x07);
  }

  /* 2 D) add displacement, if needed */
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

  io_generated_code->code_size = code_head;
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_rex_compute_2reg( libxsmm_generated_code*     io_generated_code,
                                               const unsigned int          i_instr,
                                               const unsigned int          i_reg_number_rm,
                                               const unsigned int          i_reg_number_reg )
{
  unsigned int code_head = io_generated_code->code_size;
  unsigned char* code    = (unsigned char *)io_generated_code->generated_code;
  /* prefix and op-code tables */
  unsigned char tbl_prefix[4] = {0x00, 0x66, 0xf3, 0xf2};
  unsigned char tbl_opext[2] = {0x38, 0x3a};
  unsigned int prefix_idx = ((i_instr & 0x00070000) >> 16) - 4;
  unsigned int opext = (i_instr & 0x00003000) >> 12;
  unsigned int opext_idx = (i_instr & 0x00001000) >> 12;

  /* check if we have enough code buffer space left */
  if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
    return;
  }

  /* A): writing prefixes */
  /* operand size overwrite prefix */
  if ( (i_instr & 0x0000c000) == 0x00004000 ) {
    code[code_head++] = 0x66;
  }
   /* instruction prefix */
  if ( prefix_idx != 0 ) {
    code[code_head++] = tbl_prefix[prefix_idx];
  }
  /* REX prefix */
  if ( (i_reg_number_rm > 7) || (i_reg_number_reg > 7) || ( (i_instr & 0x02000000) == 0x02000000 ) ) {
    /* some instructions have an OP code to encode the register and skip the modrm byte, then the B field is used */
    if ( (i_instr & 0x01000000) == 0x01000000 ) {
      /* B is used and X is unused */
      code[code_head  ]  = (unsigned char)(( i_reg_number_reg > 7 ) ? 0x01 : 0x00);
    } else {
      /* R */
      code[code_head  ]  = (unsigned char)(( i_reg_number_reg > 7 ) ? 0x04 : 0x00);
      /* B is used and X is unused */
      code[code_head  ] |= (unsigned char)(( i_reg_number_rm > 7 ) ? 0x01 : 0x00);
    }
    /* W */
    code[code_head  ] |= (unsigned char)(( (i_instr & 0x00800000) == 0x00800000 ) ? 0x08 : 0x00);
     /* start of REX prefix */
    code[code_head++] |= 0x40;
  }

  /* B): opcode extensions */
  if ( opext > 0 ) {
    code[code_head++] = 0x0f;
    if ( opext > 1 ) {
      code[code_head++] = tbl_opext[opext_idx];
    }
  }

  /* C) writing lowest op code byte */
  code[code_head++] = (unsigned char) (i_instr & 0x000000ff);

  /* D) setting modrm, we are in reg-only addressing mode */
  /* some instructions have an OP code to encode the register and skip the modrm byte */
  if ( (i_instr & 0x01000000) == 0x01000000 ) {
    code[code_head-1] |= (unsigned char)(i_reg_number_reg & 0x07);
  } else {
    code[code_head  ]  = (unsigned char)0xc0;
    code[code_head  ] |= (unsigned char)(((unsigned char)(i_reg_number_reg << 3)) & 0x38);
    code[code_head++] |= (unsigned char)(((unsigned char) i_reg_number_rm)       & 0x07);
  }

  io_generated_code->code_size = code_head;
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

  /* check if we have enough code buffer space left */
  if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
    return;
  }

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

  /* 1 B) determining if a force zero displacement is needed */
  if ( ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_RBP) || (i_gp_reg_base == LIBXSMM_X86_GP_REG_R13) ) && (i_displacement == 0) ) {
    l_forced_zdisp8 = 1;
  } else {
    l_forced_zdisp8 = 0;
  }

  /* 2nd phase: encoding */
  /* 2 A): writing an instruction template into the byte stream */
  /* @TODO, we right now only encode 3byte VEX */
  /* const VEX prefix */
  code[vexp ] = 0xc4;
  /* p0-op based on instruction value - this is the MMMM field, upper two bits are reserved to be 00 */
  code[p0   ] = (unsigned char)((i_vec_instr >> 12) & 0x0f);
  /* W-bit and PP prefix */
  code[p1   ] = (unsigned char)((i_vec_instr >> 16) & 0x83);
  /* we are just copying over the OP-code */
  code[op   ] = (unsigned char) i_vec_instr;

  /* 2 B) filling the missing prefix bits based on table look ups */
  /* R */
  code[p0   ] |= (unsigned char)(( i_vec_reg_number_dst < 8 ) ? 0x80 : 0x00);
  /* vvvv and V' */
  code[p1   ] |= (unsigned char)((i_vec_reg_number_src < 16) ? tbl_vex_vvvv[i_vec_reg_number_src] : tbl_vex_vvvv[0]);
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
    /* adjust code head*/
    code_head += 5;
  }

  /* 2 D) add displacement, if needed */
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

  /* check if we have enough code buffer space left */
  if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
    return;
  }

  /* encoding */
  /* A): writing an instruction template into the byte stream */
  /* const VEX prefix */
  code[vexp ] = 0xc4;
  /* p0-op based on instruction value - this is the MMMM field, upper two bits are reserved to be 00 */
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
  code[p1   ] |= (unsigned char)((i_vec_reg_number_1 < 16) ? tbl_vex_vvvv[i_vec_reg_number_1] : tbl_vex_vvvv[0]);
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
  unsigned char tbl_disp8divbcst[4] = {0x04, 0x08, 0x02, 0x00};
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

  /* check if all register args are in bound */
  if ( (i_vec_reg_number_dst > 31) || (i_vec_reg_number_src > 31) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_REGNUM);
    return;
  }
  /* check if we have enough code buffer space left */
  if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
    return;
  }

  /* 1st phase: let's compute some static information before starting the
     encoding process */
  /* 1 A) handling EVEX compressed displacement */
  if ( i_use_broadcast ) {
    l_wbit     = (unsigned char)((i_vec_instr >> 23) & 0x3);
    l_disp8div = tbl_disp8divbcst[ l_wbit ];
  } else {
    /* read initial VL=512 calibrated disp8div look up */
    l_disp8div_idx = (unsigned char)((i_vec_instr >> 8) & 0x07);
    /* check we need to adjust because of VL */
    if ( (unsigned char)((i_vec_instr >> 8) & 0x08) == 8 ) {
      /* Bit 11 is set: do not adjust depending on VL */
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

  /* 1 C) determining if a force zero displacement is needed */
  if ( ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_RBP) || (i_gp_reg_base == LIBXSMM_X86_GP_REG_R13) ) && (i_displacement == 0) ) {
    l_forced_zdisp8 = 1;
  } else {
    l_forced_zdisp8 = 0;
  }

  /* 2nd phase: encoding */
  /* 2 A): writing an instruction template into the byte stream */
  /* const EVEX prefix */
  code[evexp] = 0x62;
  /* p0-op based on instruction value - this is the MMMM field, upper two bits are reserved to be 00 */
  code[p0   ] = (unsigned char)((i_vec_instr >> 12) & 0x0f);
  /* W-bit and PP prefix */
  code[p1   ] = (unsigned char)((i_vec_instr >> 16) & 0x87);
  /* the fourth prefix byte needs to be compute, let's set it to 0 for now */
  code[p2   ] = 0x00;
  /* we are just copying over the OP-code */
  code[op   ] = (unsigned char) i_vec_instr;

  /* 2 B) filling the missing prefix bits based on table look ups */
  /* R and R' */
  assert(i_vec_reg_number_src < 32 && i_vec_reg_number_dst < 32);
  code[p0   ] |= (unsigned char)tbl_evex_RRp[i_vec_reg_number_dst];
  /* vvvv and V' */
  code[p1   ] |= (unsigned char)tbl_evex_vvvv[i_vec_reg_number_src];
  /* in case of gather scatter the V' field is used to extend the idx field for SIB to 32 registers */
  if ( (((i_vec_instr >> 24) & 0x2) == 0x2) ) {
    code[p2   ] |= (unsigned char)((l_reg_idx < 16 ) ? tbl_evex_vp[l_reg_idx] : tbl_evex_vp[0]);
  } else {
    code[p2   ] |= (unsigned char)tbl_evex_vp[i_vec_reg_number_src];
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
    /* adjust code head*/
    code_head += 6;
  }

  /* 2 D) add displacement, if needed */
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

  /* check if all register args are in bound */
  if ( (i_vec_reg_number_0 > 31) || (i_vec_reg_number_1 > 31) || (i_vec_reg_number_2 > 31) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ILLEGAL_REGNUM);
    return;
  }
  /* check if we have enough code buffer space left */
  if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
    return;
  }

  /* encoding */
  /* A): writing an instruction template into the byte stream */
  /* const EVEX prefix */
  code[evexp] = 0x62;
  /* p0-op based on instruction value - this is the MMMM field, upper two bits are reserved to be 00 */
  code[p0   ] = (unsigned char)((i_vec_instr >> 12) & 0x0f);
  /* W-bit and PP prefix */
  code[p1   ] = (unsigned char)((i_vec_instr >> 16) & 0x87);
  /* the fourth prefix byte needs to be compute, let's set it to 0 for now */
  code[p2   ] = 0x00;
  /* we are just copying over the OP-code */
  code[op   ] = (unsigned char) i_vec_instr;

  /* B) filling the missing prefix bits based on table look ups */
  /* R and R' */
  assert(i_vec_reg_number_0 < 32 && i_vec_reg_number_1 < 32 && i_vec_reg_number_2 < 32);
  code[p0   ] |= (unsigned char)tbl_evex_RRp[i_vec_reg_number_2];
  /* B and X */
  code[p0   ] |= (unsigned char)tbl_evex_BX[i_vec_reg_number_0];
  /* vvvv and V' */
  code[p1   ] |= (unsigned char)tbl_evex_vvvv[i_vec_reg_number_1];
  code[p2   ] |= (unsigned char)tbl_evex_vp[i_vec_reg_number_1];
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
      fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: unexpected instruction number: 0x%08x\n", i_vmove_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
}

  /* select the code generator REX/VEX/EVEX */
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) &&
       (io_generated_code->code_type > 1) ) {
    libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;

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

    /* check for gather */
    if ( (((i_vmove_instr >> 24) & 0x2) == 0x2) ) {
      if (i_reg_idx > 15) {
        fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: SIB addressing mode is required for instruction number: 0x%08x\n", i_vmove_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
      if ( (i_vec_reg_mask_0 == i_vec_reg_number_0) || (i_reg_idx == i_vec_reg_number_0) || (i_reg_idx == i_vec_reg_mask_0) ) {
        fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: same register names cannot be used multiple times: 0x%08x\n", i_vmove_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  } else {
    /* general encoder error */
    fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: GENERAL ERROR\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_move( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_instruction_set,
                                       const unsigned int      i_vmove_instr,
                                       const unsigned int      i_gp_reg_base,
                                       const unsigned int      i_reg_idx,
                                       const unsigned int      i_scale,
                                       const int               i_displacement,
                                       const char              i_vector_name,
                                       const unsigned int      i_vec_reg_number_0,
                                       const unsigned int      i_mask_reg_number,
                                       const unsigned int      i_use_zero_masking,
                                       const unsigned int      i_is_store )
{

  /* check for correct streaming stores */
  if ( (i_is_store == 0) && ( (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVNTPD) ||
                              (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVNTPS) ||
                              (i_vmove_instr == LIBXSMM_X86_INSTR_VMOVNTDQ)   )) {
    fprintf(stderr, "libxsmm_instruction_vec_move: streaming stores are only available when setting storing option to true!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* check that we are not masking 'y' */
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX) && (i_mask_reg_number != 0) ) {
    fprintf(stderr, "libxsmm_instruction_vec_move: Masking is only available for AVX512!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* check zero masking */
  if ( (i_use_zero_masking != 0) && (i_mask_reg_number != 0) && (i_is_store != 0) ) {
    fprintf(stderr, "libxsmm_instruction_vec_move: zero-masked store cannot operate on memory destination!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* select the code generator REX/VEX/EVEX */
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_vmove_instr;

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
      case LIBXSMM_X86_INSTR_MOVAPD:
        l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_MOVAPD_LD : LIBXSMM_X86_INSTR_MOVAPD_ST;
        break;
      case LIBXSMM_X86_INSTR_MOVUPD:
        l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_MOVUPD_LD : LIBXSMM_X86_INSTR_MOVUPD_ST;
        break;
      case LIBXSMM_X86_INSTR_MOVAPS:
        l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_MOVAPS_LD : LIBXSMM_X86_INSTR_MOVAPS_ST;
        break;
      case LIBXSMM_X86_INSTR_MOVUPS:
        l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_MOVUPS_LD : LIBXSMM_X86_INSTR_MOVUPS_ST;
        break;
      case LIBXSMM_X86_INSTR_MOVSD:
        l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_MOVSD_LD : LIBXSMM_X86_INSTR_MOVSD_ST;
        break;
      case LIBXSMM_X86_INSTR_MOVSS:
        l_vmove_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_MOVSS_LD : LIBXSMM_X86_INSTR_MOVSS_ST;
        break;
      default:
        l_vmove_instr = i_vmove_instr;
        break;
    }

    if ( ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) && ( l_vmove_instr == LIBXSMM_X86_INSTR_VBROADCASTSD ) ) {
      l_vmove_instr = LIBXSMM_X86_INSTR_VBROADCASTSD_VEX;
    }

    /* uses short-cut encoder */
    libxsmm_x86_instruction_vec_compute_mem_1reg_mask( io_generated_code, l_vmove_instr, i_vector_name,
                                                       i_gp_reg_base, i_reg_idx, i_scale, i_displacement, 0,
                                                       i_vec_reg_number_0, i_mask_reg_number, i_use_zero_masking );
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

    if ( (i_instruction_set >= LIBXSMM_X86_AVX512_VL128_SKX) &&
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
                                                             const unsigned int      i_imm8 ) {
  if ( (libxsmm_x86_instruction_vec_is_hybrid( i_vec_instr )  == 0) &&
       (libxsmm_x86_instruction_vec_is_regonly( i_vec_instr ) == 0)    ) {
    fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: unexpected instruction number: 0x%08x\n", i_vec_instr);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* check that we are not masking 'y' */
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX) && (i_mask_reg_number != 0) ) {
    fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: Masking is only available for AVX512!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* select the code generator REX/VEX/EVEX */
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_encoder; /* 2=EVEX, 1=VEX, 0=REX */
    unsigned int l_encoder_arch = 2;
    unsigned int l_encoder_instr = ((i_vec_instr >> 30) & 0x03);
    unsigned int l_reg_number_src0 = 0;
    unsigned int l_reg_number_src1 = 0;
    unsigned int l_reg_number_dst = 0;

    /* determine encoder */
    if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
      l_encoder_arch = 0;
    }
    else if ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
      l_encoder_arch = 1;
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
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: In case of a 2 src operand instruction (0x%08x), i_reg_number_src1 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
      l_reg_number_src1 = 0;
    } else if ( ((i_vec_instr >> 28) & 3) == 1 ) {
      if ( i_reg_number_src0 != LIBXSMM_X86_VEC_REG_UNDEF ) {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: In case of a 1 src operand instruction (0x%08x), i_reg_number_src0 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
      l_reg_number_src0 = 0;
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
      } else if ( ((i_vec_instr >> 28) & 0x3) == 0x1 )  {
        l_reg_number_src0 = i_reg_number_dst;
        l_reg_number_dst = ((i_vec_instr >> 20) & 0x07);
      } else {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: In case of a op-code modrm/reg extended instruction (0x%08x), i_reg_number_src1 or i_reg_number_src0 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
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
      libxsmm_x86_instruction_rex_compute_2reg( io_generated_code, i_vec_instr,
            l_reg_number_src0, l_reg_number_dst );
    }

    /* add imm if needed */
    if ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) {
      if ( i_imm8 != LIBXSMM_X86_IMM_UNDEF ) {
        unsigned char* code = (unsigned char *) io_generated_code->generated_code;
        code[io_generated_code->code_size++] = (unsigned char)i_imm8;
      } else {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8: imm8 required by instr, but LIBXSMM_X86_IMM_UNDEF was provided!\n");
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
    }
  } else {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;
    char l_instr_name[16];
    unsigned int l_imm8 = (unsigned int)i_imm8;
    libxsmm_get_x86_instr_name( i_vec_instr, l_instr_name, 15 );

    /* build vXYZpd/ps/sd/ss instruction pure register use*/
    if ( io_generated_code->arch > LIBXSMM_X86_SSE42 ) {
      if ( ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) && (i_imm8 != LIBXSMM_X86_IMM_UNDEF) ) {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s $%u, %%%%%cmm%u, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, l_imm8, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%u, %%%cmm%u, %%%cmm%u, %%%cmm%u\n", l_instr_name, l_imm8, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
        }
      }
    } else {
      if ( ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) && (i_imm8 != LIBXSMM_X86_IMM_UNDEF) ) {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s $%u, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, l_imm8, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_dst );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%u, %%%cmm%u, %%%cmm%u\n", l_instr_name, l_imm8, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_dst );
        }
      } else {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_dst );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %%%cmm%u\n", l_instr_name, i_vector_name, i_reg_number_src0, i_vector_name, i_reg_number_dst );
        }
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
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
                                                    const unsigned int      i_imm8 ) {
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
                                                             const unsigned int      i_imm8 ) {
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
                                                    const unsigned int      i_imm8 ) {
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, i_vec_instr, i_vector_name,
                                                           i_reg_number_src0, LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
                                                           0, 0, 0, i_imm8 );
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_1reg_imm8( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_instr,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_reg_number_dst,
                                                    const unsigned int      i_imm8 ) {
  libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8 ( io_generated_code, i_vec_instr, i_vector_name,
                                                           LIBXSMM_X86_VEC_REG_UNDEF, LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
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
                                                             const unsigned int      i_imm8 )
{
  if ( (libxsmm_x86_instruction_vec_is_hybrid( i_vec_instr )     == 0) &&
       (libxsmm_x86_instruction_vec_is_regmemonly( i_vec_instr ) == 0)    ) {
    fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: unexpected instruction number: 0x%08x\n", i_vec_instr);
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* check that we are not masking 'y' */
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX) && (i_mask_reg_number != 0) ) {
    fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: Masking is only available for AVX512!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }

  /* select the code generator REX/VEX/EVEX */
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_encoder; /* 2=EVEX, 1=VEX, 0=REX */
    unsigned int l_encoder_arch = 2;
    unsigned int l_encoder_instr = ((i_vec_instr >> 30) & 0x03);
    unsigned int l_reg_number_src1;
    unsigned int l_reg_number_dst = i_reg_number_dst;

    /* determine encoder */
    if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
      l_encoder_arch = 0;
    }
    else if ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
      l_encoder_arch = 1;
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
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: In case of a 1 src operand instruction (0x%08x), i_reg_number_src1 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
      l_reg_number_src1 = 0;
    } else {
      l_reg_number_src1 = i_reg_number_src1;
    }

    /* check that we have an UNDEF for both vec reg operands */
    if ( ((i_vec_instr >> 28) & 3) == 1 ) {
      if ( (i_reg_number_src1 != LIBXSMM_X86_VEC_REG_UNDEF) || (i_reg_number_dst != LIBXSMM_X86_VEC_REG_UNDEF) ) {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: In case of a 0 src operand instruction (0x%08x), i_reg_number_src1 and i_reg_number_dst needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
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
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: In case of a op-code modrm/reg extended instruction (0x%08x), i_reg_number_src1 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_vec_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
    }

    /* encode main instruction */
    if ( l_encoder == 2 ) {
      libxsmm_x86_simd_name l_simd_name = LIBXSMM_X86_SIMD_NAME_XMM;

      /* check for gather/scatter */
      if ( (((i_vec_instr >> 24) & 0x2) == 0x2) ) {
        if (i_gp_reg_idx > 32) {
          fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: SIB addressing mode is required for instruction number: 0x%08x\n", i_vec_instr);
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
        }
        if ( (i_mask_rnd_exp_cntl != 0) || (0 == i_mask_reg_number) ) {
          fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: merge masking with a valid mask registers (>k0) is required for instruction number: 0x%08x\n", i_vec_instr);
          LIBXSMM_EXIT_ERROR(io_generated_code);
          return;
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
      libxsmm_x86_instruction_rex_compute_1reg_mem( io_generated_code, i_vec_instr,
            i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, l_reg_number_dst );
    }

    /* add imm if needed */
    if ( ((i_vec_instr >> 16) & 0x08) == 0x08 ) {
      if ( i_imm8 != LIBXSMM_X86_IMM_UNDEF ) {
        unsigned char* code = (unsigned char *) io_generated_code->generated_code;
        code[io_generated_code->code_size++] = (unsigned char)i_imm8;
      } else {
        fprintf(stderr, "libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8: imm8 required by instr, but LIBXSMM_X86_IMM_UNDEF was provided!\n");
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
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

    if ( io_generated_code->arch > LIBXSMM_X86_SSE42 ) {
      if (l_single_precision == 0) {
        LIBXSMM_SNPRINTF( l_broadcast, 7, "1to8" );
      } else {
        LIBXSMM_SNPRINTF( l_broadcast, 7, "1to16" );
      }

      /* build vXYZpd/ps/sd/ss instruction pure register use*/
      if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF ) {
        if ( io_generated_code->code_type == 0 ) {
          if (i_use_broadcast != 0) {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s)%%{%s%%}, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_broadcast, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
          } else {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s), %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
          }
        } else {
          if (i_use_broadcast != 0) {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s) {%s}, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_broadcast, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
          } else {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
          }
        }
      } else {
        libxsmm_get_x86_gp_reg_name( i_gp_reg_idx, l_gp_reg_idx, 3 );
        if ( io_generated_code->code_type == 0 ) {
          if (i_use_broadcast != 0) {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%%s,%u)%%{%s%%}, %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, l_broadcast, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
          } else {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s,%%%%%s,%u), %%%%%cmm%u, %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
          }
        } else {
          if (i_use_broadcast != 0) {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u) {%s}, %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, l_broadcast, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
          } else {
            l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s,%%%s,%u), %%%cmm%u, %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, l_gp_reg_idx, i_scale, i_vector_name, i_reg_number_src1, i_vector_name, i_reg_number_dst );
          }
        }
      }
    } else {
      if ( io_generated_code->code_type == 0 ) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %i(%%%%%s), %%%%%cmm%u\\n\\t\"\n", l_instr_name, i_displacement, l_gp_reg_base, i_vector_name, i_reg_number_dst );
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%%cmm%u\n", l_instr_name, i_displacement, l_gp_reg_base, i_vector_name, i_reg_number_dst );
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
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
                                                        const unsigned int      i_imm8 ) {
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
                                                             const unsigned int      i_imm8 ) {
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
                                                        const unsigned int      i_imm8 ) {
  libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( io_generated_code, i_vec_instr, i_vector_name,
                                                          i_gp_reg_base, i_gp_reg_idx, i_scale, i_displacement, i_use_broadcast,
                                                          LIBXSMM_X86_VEC_REG_UNDEF, i_reg_number_dst,
                                                          0, 0, i_imm8 );
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
  if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL128_SKX) {
    if ( i_use_masking != 0 ) {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_vmove_instr,
                                        i_gp_reg_base, i_reg_idx, i_scale, i_displacement,
                                        i_vector_name, i_vec_reg_number_0, i_mask_reg_number, (i_is_store != 0) ? 0 : 1, i_is_store );
    } else {
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, i_vmove_instr,
                                        i_gp_reg_base, i_reg_idx, i_scale, i_displacement,
                                        i_vector_name, i_vec_reg_number_0, 0, (i_is_store != 0) ? 0 : 1, i_is_store );
    }
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) /*&& (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX)*/) {
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
  switch ( i_prefetch_instr ) {
    case LIBXSMM_X86_INSTR_PREFETCHT0:
    case LIBXSMM_X86_INSTR_PREFETCHT1:
    case LIBXSMM_X86_INSTR_PREFETCHT2:
    case LIBXSMM_X86_INSTR_PREFETCHNTA:
    case LIBXSMM_X86_INSTR_PREFETCHW:
    case LIBXSMM_X86_INSTR_CLDEMOTE:
    case LIBXSMM_X86_INSTR_CLFLUSH:
    case LIBXSMM_X86_INSTR_CLFLUSHOPT:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_prefetch: Unknown instruction type: 0x%08x\n", i_prefetch_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_reg_op_ext = 0;

    /* check if we have op-code extension in modrm/reg and correct operand count */
#if 0 /* dead condition */
    if ( ((i_prefetch_instr >> 24) & 0x04 ) == 0x04 )
#endif
    {
#if 0 /* dead condition */
      if ( ((i_prefetch_instr >> 28) & 0x3) == 0x1 )
#endif
      {
        l_reg_op_ext = ((i_prefetch_instr >> 20) & 0x07);
      }
#if 0 /* dead condition */
      else {
        fprintf(stderr, "libxsmm_x86_instruction_prefetch: Instruction (0x%08x) must have only one operand!\n", i_prefetch_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
#endif
    }
#if 0 /* dead condition */
    else {
      fprintf(stderr, "libxsmm_x86_instruction_prefetch: Instruction (0x%08x) has no op-code modrm/reg extension!\n", i_prefetch_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }
#endif
    libxsmm_x86_instruction_rex_compute_1reg_mem ( io_generated_code,
                                                   i_prefetch_instr, i_gp_reg_base,
                                                   i_gp_reg_idx, i_scale, i_displacement, l_reg_op_ext );
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
                                      const unsigned int      i_alu_instr,
                                      const unsigned int      i_gp_reg_base,
                                      const unsigned int      i_gp_reg_idx,
                                      const unsigned int      i_scale,
                                      const int               i_displacement,
                                      const unsigned int      i_gp_reg_number,
                                      const unsigned int      i_is_store ) {
  switch ( i_alu_instr ) {
    case LIBXSMM_X86_INSTR_MOVQ:
    case LIBXSMM_X86_INSTR_MOVD:
    case LIBXSMM_X86_INSTR_MOVW:
    case LIBXSMM_X86_INSTR_MOVB:
#if 0
    case LIBXSMM_X86_INSTR_MOVQ_LD:
    case LIBXSMM_X86_INSTR_MOVD_LD:
    case LIBXSMM_X86_INSTR_MOVW_LD:
    case LIBXSMM_X86_INSTR_MOVB_LD:
#endif
    case LIBXSMM_X86_INSTR_MOVQ_ST:
    case LIBXSMM_X86_INSTR_MOVD_ST:
    case LIBXSMM_X86_INSTR_MOVW_ST:
    case LIBXSMM_X86_INSTR_MOVB_ST:
    case LIBXSMM_X86_INSTR_LEAW:
    case LIBXSMM_X86_INSTR_LEAD:
    case LIBXSMM_X86_INSTR_LEAQ:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_alu_mem: Unknown instruction type: 0x%08x\n", i_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 )
  {
    unsigned int l_alu_instr = i_alu_instr;

    switch (i_alu_instr) {
      case LIBXSMM_X86_INSTR_MOVQ:
        l_alu_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_MOVQ_LD : LIBXSMM_X86_INSTR_MOVQ_ST;
        break;
      case LIBXSMM_X86_INSTR_MOVD:
        l_alu_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_MOVD_LD : LIBXSMM_X86_INSTR_MOVD_ST;
        break;
      case LIBXSMM_X86_INSTR_MOVW:
        l_alu_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_MOVW_LD : LIBXSMM_X86_INSTR_MOVW_ST;
        break;
      case LIBXSMM_X86_INSTR_MOVB:
        l_alu_instr = (i_is_store == 0) ? LIBXSMM_X86_INSTR_MOVB_LD : LIBXSMM_X86_INSTR_MOVB_ST;
        break;
    }

    libxsmm_x86_instruction_rex_compute_1reg_mem ( io_generated_code,
                                                   l_alu_instr, i_gp_reg_base,
                                                   i_gp_reg_idx, i_scale, i_displacement, i_gp_reg_number );
  } else {
    /* TODO: */
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_alu_imm( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_alu_instr,
                                      const unsigned int      i_gp_reg_number,
                                      const long long         i_immediate ) {
  switch ( i_alu_instr ) {
    case LIBXSMM_X86_INSTR_ADDQ:
    case LIBXSMM_X86_INSTR_ADDB_RM_IMM8:
    case LIBXSMM_X86_INSTR_ADDW_RM_IMM16:
    case LIBXSMM_X86_INSTR_ADDD_RM_IMM32:
    case LIBXSMM_X86_INSTR_ADDQ_RM_IMM32:
    case LIBXSMM_X86_INSTR_ANDQ:
    case LIBXSMM_X86_INSTR_ANDB_RM_IMM8:
    case LIBXSMM_X86_INSTR_ANDW_RM_IMM16:
    case LIBXSMM_X86_INSTR_ANDD_RM_IMM32:
    case LIBXSMM_X86_INSTR_ANDQ_RM_IMM32:
    case LIBXSMM_X86_INSTR_CMPQ:
    case LIBXSMM_X86_INSTR_CMPB_RM_IMM8:
    case LIBXSMM_X86_INSTR_CMPW_RM_IMM16:
    case LIBXSMM_X86_INSTR_CMPD_RM_IMM32:
    case LIBXSMM_X86_INSTR_CMPQ_RM_IMM32:
    case LIBXSMM_X86_INSTR_IMUL:
    case LIBXSMM_X86_INSTR_IMULW_IMM16:
    case LIBXSMM_X86_INSTR_IMULD_IMM32:
    case LIBXSMM_X86_INSTR_IMULQ_IMM32:
    case LIBXSMM_X86_INSTR_MOVQ:
    case LIBXSMM_X86_INSTR_MOVB_RM_IMM8:
    case LIBXSMM_X86_INSTR_MOVW_RM_IMM16:
    case LIBXSMM_X86_INSTR_MOVD_RM_IMM32:
    case LIBXSMM_X86_INSTR_MOVQ_RM_IMM32:
    case LIBXSMM_X86_INSTR_ORB_RM_IMM8:
    case LIBXSMM_X86_INSTR_ORW_RM_IMM16:
    case LIBXSMM_X86_INSTR_ORD_RM_IMM32:
    case LIBXSMM_X86_INSTR_ORQ_RM_IMM32:
    case LIBXSMM_X86_INSTR_SHLQ:
    case LIBXSMM_X86_INSTR_SHLB_RM_IMM8:
    case LIBXSMM_X86_INSTR_SHLW_RM_IMM8:
    case LIBXSMM_X86_INSTR_SHLD_RM_IMM8:
    case LIBXSMM_X86_INSTR_SHLQ_RM_IMM8:
    case LIBXSMM_X86_INSTR_SARQ:
    case LIBXSMM_X86_INSTR_SARB_RM_IMM8:
    case LIBXSMM_X86_INSTR_SARW_RM_IMM8:
    case LIBXSMM_X86_INSTR_SARD_RM_IMM8:
    case LIBXSMM_X86_INSTR_SARQ_RM_IMM8:
    case LIBXSMM_X86_INSTR_SHRQ:
    case LIBXSMM_X86_INSTR_SHRB_RM_IMM8:
    case LIBXSMM_X86_INSTR_SHRW_RM_IMM8:
    case LIBXSMM_X86_INSTR_SHRD_RM_IMM8:
    case LIBXSMM_X86_INSTR_SHRQ_RM_IMM8:
    case LIBXSMM_X86_INSTR_SUBQ:
    case LIBXSMM_X86_INSTR_SUBB_RM_IMM8:
    case LIBXSMM_X86_INSTR_SUBW_RM_IMM16:
    case LIBXSMM_X86_INSTR_SUBD_RM_IMM32:
    case LIBXSMM_X86_INSTR_SUBQ_RM_IMM32:
    case LIBXSMM_X86_INSTR_XORB_RM_IMM8:
    case LIBXSMM_X86_INSTR_XORW_RM_IMM16:
    case LIBXSMM_X86_INSTR_XORD_RM_IMM32:
    case LIBXSMM_X86_INSTR_XORQ_RM_IMM32:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_alu_imm: Unknown instruction type: 0x%08x\n", i_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_reg_number_dst = 0;
    unsigned int l_reg_number_src0 = i_gp_reg_number;
    unsigned int l_alu_instr = i_alu_instr;

    switch (i_alu_instr) {
      case LIBXSMM_X86_INSTR_ADDQ:
        l_alu_instr = LIBXSMM_X86_INSTR_ADDQ_RM_IMM32;
        break;
      case LIBXSMM_X86_INSTR_ANDQ:
        l_alu_instr = LIBXSMM_X86_INSTR_ANDQ_RM_IMM32;
        break;
      case LIBXSMM_X86_INSTR_CMPQ:
        l_alu_instr = LIBXSMM_X86_INSTR_CMPQ_RM_IMM32;
        break;
      case LIBXSMM_X86_INSTR_IMUL:
        l_alu_instr = LIBXSMM_X86_INSTR_IMULQ_IMM32;
        break;
      case LIBXSMM_X86_INSTR_MOVQ:
        l_alu_instr = LIBXSMM_X86_INSTR_MOVQ_RM_IMM32;
        break;
      case LIBXSMM_X86_INSTR_SARQ:
        l_alu_instr = LIBXSMM_X86_INSTR_SARQ_RM_IMM8;
        break;
      case LIBXSMM_X86_INSTR_SHLQ:
        l_alu_instr = LIBXSMM_X86_INSTR_SHLQ_RM_IMM8;
        break;
      case LIBXSMM_X86_INSTR_SHRQ:
        l_alu_instr = LIBXSMM_X86_INSTR_SHRQ_RM_IMM8;
        break;
      case LIBXSMM_X86_INSTR_SUBQ:
        l_alu_instr = LIBXSMM_X86_INSTR_SUBQ_RM_IMM32;
        break;
    }

    /* check if we have op-code extension in modrm/reg */
    if ( ((l_alu_instr >> 24) & 0x04 ) == 0x04 ) {
#if 0 /* dead condition */
      if ( ((l_alu_instr >> 28) & 0x3) == 0x1 )
#endif
      {
        l_reg_number_dst = ((l_alu_instr >> 20) & 0x07);
      }
#if 0 /* dead condition */
      else {
        fprintf(stderr, "libxsmm_x86_instruction_alu_imm: In case of a op-code modrm/reg extended instruction (0x%08x) only one register operand is allowed!\n", l_alu_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
#endif
    }

    /* TODO: fix! for IMUL we have a ternay instrucation */
    if ( l_alu_instr == LIBXSMM_X86_INSTR_IMULW_IMM16 ||
         l_alu_instr == LIBXSMM_X86_INSTR_IMULD_IMM32 ||
         l_alu_instr == LIBXSMM_X86_INSTR_IMULQ_IMM32 ) {
      l_reg_number_dst = i_gp_reg_number;
    }

    /* generate the main instruction */
    libxsmm_x86_instruction_rex_compute_2reg( io_generated_code, l_alu_instr,
            l_reg_number_src0, l_reg_number_dst );

    /* copy the immediate after the insturuction */
    if ( (l_alu_instr & 0x00080000 ) == 0x00080000 ) {
      unsigned char* code = (unsigned char *) io_generated_code->generated_code;
      unsigned int l_imm_bytes = 1 << ( (l_alu_instr >> 8) & 0x3 );
      unsigned int l_immediate = (unsigned int)i_immediate;
      unsigned int l_i = 0;
      for ( l_i = 0; l_i < l_imm_bytes; ++l_i ) {
        code[io_generated_code->code_size++] = (unsigned char)l_immediate;
        l_immediate = l_immediate >> 8;
      }
    } else {
      fprintf(stderr, "libxsmm_x86_instruction_alu_imm: Instruction (0x%08x) is not an imm-instruction!\n", l_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }
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
                                          const long long         i_immediate ) {
  switch ( i_alu_instr ) {
    case LIBXSMM_X86_INSTR_MOVQ:
    case LIBXSMM_X86_INSTR_MOVB_R_IMM8:
    case LIBXSMM_X86_INSTR_MOVW_R_IMM16:
    case LIBXSMM_X86_INSTR_MOVD_R_IMM32:
    case LIBXSMM_X86_INSTR_MOVQ_R_IMM64:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_alu_imm_i64: Unknown instruction type: 0x%08x\n", i_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_alu_instr = i_alu_instr;

    switch (i_alu_instr) {
      case LIBXSMM_X86_INSTR_MOVQ:
        l_alu_instr = LIBXSMM_X86_INSTR_MOVQ_R_IMM64;
        break;
    }

    /* generate the main instruction */
    libxsmm_x86_instruction_rex_compute_2reg( io_generated_code, l_alu_instr,
            0, i_gp_reg_number );

    /* copy the immediate after the insturuction */
    if ( (l_alu_instr & 0x00080000 ) == 0x00080000 ) {
      unsigned char* code = (unsigned char *) io_generated_code->generated_code;
      unsigned int l_imm_bytes = 1 << ( (l_alu_instr >> 8) & 0x3 );
      size_t l_immediate = i_immediate;
      unsigned int l_i = 0;
      for ( l_i = 0; l_i < l_imm_bytes; ++l_i ) {
        code[io_generated_code->code_size++] = (unsigned char)l_immediate;
        l_immediate = (size_t)((size_t)l_immediate >> (size_t)8);
      }
    } else {
      fprintf(stderr, "libxsmm_x86_instruction_alu_imm_i64: Instruction (0x%08x) is not an imm-instruction!\n", l_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }
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
  switch ( i_alu_instr ) {
    case LIBXSMM_X86_INSTR_ADDQ:
    case LIBXSMM_X86_INSTR_ADDB_RM_R:
    case LIBXSMM_X86_INSTR_ADDW_RM_R:
    case LIBXSMM_X86_INSTR_ADDD_RM_R:
    case LIBXSMM_X86_INSTR_ADDQ_RM_R:
    case LIBXSMM_X86_INSTR_ADDB_R_RM:
    case LIBXSMM_X86_INSTR_ADDW_R_RM:
    case LIBXSMM_X86_INSTR_ADDD_R_RM:
    case LIBXSMM_X86_INSTR_ADDQ_R_RM:
    case LIBXSMM_X86_INSTR_ANDQ:
    case LIBXSMM_X86_INSTR_ANDB_RM_R:
    case LIBXSMM_X86_INSTR_ANDW_RM_R:
    case LIBXSMM_X86_INSTR_ANDD_RM_R:
    case LIBXSMM_X86_INSTR_ANDQ_RM_R:
    case LIBXSMM_X86_INSTR_ANDB_R_RM:
    case LIBXSMM_X86_INSTR_ANDW_R_RM:
    case LIBXSMM_X86_INSTR_ANDD_R_RM:
    case LIBXSMM_X86_INSTR_ANDQ_R_RM:
    case LIBXSMM_X86_INSTR_CMOVAW:
    case LIBXSMM_X86_INSTR_CMOVAD:
    case LIBXSMM_X86_INSTR_CMOVAQ:
    case LIBXSMM_X86_INSTR_CMOVAEW:
    case LIBXSMM_X86_INSTR_CMOVAED:
    case LIBXSMM_X86_INSTR_CMOVAEQ:
    case LIBXSMM_X86_INSTR_CMOVBW:
    case LIBXSMM_X86_INSTR_CMOVBD:
    case LIBXSMM_X86_INSTR_CMOVBQ:
    case LIBXSMM_X86_INSTR_CMOVBEW:
    case LIBXSMM_X86_INSTR_CMOVBED:
    case LIBXSMM_X86_INSTR_CMOVBEQ:
    case LIBXSMM_X86_INSTR_CMOVEW:
    case LIBXSMM_X86_INSTR_CMOVED:
    case LIBXSMM_X86_INSTR_CMOVEQ:
    case LIBXSMM_X86_INSTR_CMOVGW:
    case LIBXSMM_X86_INSTR_CMOVGD:
    case LIBXSMM_X86_INSTR_CMOVGQ:
    case LIBXSMM_X86_INSTR_CMOVGEW:
    case LIBXSMM_X86_INSTR_CMOVGED:
    case LIBXSMM_X86_INSTR_CMOVGEQ:
    case LIBXSMM_X86_INSTR_CMOVLW:
    case LIBXSMM_X86_INSTR_CMOVLD:
    case LIBXSMM_X86_INSTR_CMOVLQ:
    case LIBXSMM_X86_INSTR_CMOVLEW:
    case LIBXSMM_X86_INSTR_CMOVLED:
    case LIBXSMM_X86_INSTR_CMOVLEQ:
    case LIBXSMM_X86_INSTR_CMOVNEW:
    case LIBXSMM_X86_INSTR_CMOVNED:
    case LIBXSMM_X86_INSTR_CMOVNEQ:
    case LIBXSMM_X86_INSTR_CMOVNOW:
    case LIBXSMM_X86_INSTR_CMOVNOD:
    case LIBXSMM_X86_INSTR_CMOVNOQ:
    case LIBXSMM_X86_INSTR_CMOVNPW:
    case LIBXSMM_X86_INSTR_CMOVNPD:
    case LIBXSMM_X86_INSTR_CMOVNPQ:
    case LIBXSMM_X86_INSTR_CMOVNSW:
    case LIBXSMM_X86_INSTR_CMOVNSD:
    case LIBXSMM_X86_INSTR_CMOVNSQ:
    case LIBXSMM_X86_INSTR_CMOVOW:
    case LIBXSMM_X86_INSTR_CMOVOD:
    case LIBXSMM_X86_INSTR_CMOVOQ:
    case LIBXSMM_X86_INSTR_CMOVPW:
    case LIBXSMM_X86_INSTR_CMOVPD:
    case LIBXSMM_X86_INSTR_CMOVPQ:
    case LIBXSMM_X86_INSTR_CMOVSW:
    case LIBXSMM_X86_INSTR_CMOVSD:
    case LIBXSMM_X86_INSTR_CMOVSQ:
    case LIBXSMM_X86_INSTR_CMPQ:
    case LIBXSMM_X86_INSTR_CMPB_RM_R:
    case LIBXSMM_X86_INSTR_CMPW_RM_R:
    case LIBXSMM_X86_INSTR_CMPD_RM_R:
    case LIBXSMM_X86_INSTR_CMPQ_RM_R:
    case LIBXSMM_X86_INSTR_CMPB_R_RM:
    case LIBXSMM_X86_INSTR_CMPW_R_RM:
    case LIBXSMM_X86_INSTR_CMPD_R_RM:
    case LIBXSMM_X86_INSTR_CMPQ_R_RM:
    case LIBXSMM_X86_INSTR_IDIVW:
    case LIBXSMM_X86_INSTR_IDIVD:
    case LIBXSMM_X86_INSTR_IDIVQ:
    case LIBXSMM_X86_INSTR_IMUL:
    case LIBXSMM_X86_INSTR_IMULW:
    case LIBXSMM_X86_INSTR_IMULD:
    case LIBXSMM_X86_INSTR_IMULQ:
    case LIBXSMM_X86_INSTR_LZCNTW:
    case LIBXSMM_X86_INSTR_LZCNTD:
    case LIBXSMM_X86_INSTR_LZCNTQ:
    case LIBXSMM_X86_INSTR_MOVB_LD:
    case LIBXSMM_X86_INSTR_MOVW_LD:
    case LIBXSMM_X86_INSTR_MOVD_LD:
    case LIBXSMM_X86_INSTR_MOVQ_LD:
    case LIBXSMM_X86_INSTR_MOVB_ST:
    case LIBXSMM_X86_INSTR_MOVW_ST:
    case LIBXSMM_X86_INSTR_MOVD_ST:
    case LIBXSMM_X86_INSTR_MOVQ_ST:
    case LIBXSMM_X86_INSTR_NEGB:
    case LIBXSMM_X86_INSTR_NEGW:
    case LIBXSMM_X86_INSTR_NEGD:
    case LIBXSMM_X86_INSTR_NEGQ:
    case LIBXSMM_X86_INSTR_NOTB:
    case LIBXSMM_X86_INSTR_NOTW:
    case LIBXSMM_X86_INSTR_NOTD:
    case LIBXSMM_X86_INSTR_NOTQ:
    case LIBXSMM_X86_INSTR_ORB_RM_R:
    case LIBXSMM_X86_INSTR_ORW_RM_R:
    case LIBXSMM_X86_INSTR_ORD_RM_R:
    case LIBXSMM_X86_INSTR_ORQ_RM_R:
    case LIBXSMM_X86_INSTR_ORB_R_RM:
    case LIBXSMM_X86_INSTR_ORW_R_RM:
    case LIBXSMM_X86_INSTR_ORD_R_RM:
    case LIBXSMM_X86_INSTR_ORQ_R_RM:
    case LIBXSMM_X86_INSTR_POPW:
    case LIBXSMM_X86_INSTR_POPQ:
    case LIBXSMM_X86_INSTR_POPW_RM:
    case LIBXSMM_X86_INSTR_POPQ_RM:
    case LIBXSMM_X86_INSTR_POPCNT:
    case LIBXSMM_X86_INSTR_POPCNTW:
    case LIBXSMM_X86_INSTR_POPCNTD:
    case LIBXSMM_X86_INSTR_POPCNTQ:
    case LIBXSMM_X86_INSTR_PUSHW:
    case LIBXSMM_X86_INSTR_PUSHQ:
    case LIBXSMM_X86_INSTR_PUSHW_RM:
    case LIBXSMM_X86_INSTR_PUSHQ_RM:
    case LIBXSMM_X86_INSTR_SUBQ:
    case LIBXSMM_X86_INSTR_SUBB_RM_R:
    case LIBXSMM_X86_INSTR_SUBW_RM_R:
    case LIBXSMM_X86_INSTR_SUBD_RM_R:
    case LIBXSMM_X86_INSTR_SUBQ_RM_R:
    case LIBXSMM_X86_INSTR_SUBB_R_RM:
    case LIBXSMM_X86_INSTR_SUBW_R_RM:
    case LIBXSMM_X86_INSTR_SUBD_R_RM:
    case LIBXSMM_X86_INSTR_SUBQ_R_RM:
    case LIBXSMM_X86_INSTR_TZCNT:
    case LIBXSMM_X86_INSTR_TZCNTW:
    case LIBXSMM_X86_INSTR_TZCNTD:
    case LIBXSMM_X86_INSTR_TZCNTQ:
    case LIBXSMM_X86_INSTR_XORB_RM_R:
    case LIBXSMM_X86_INSTR_XORW_RM_R:
    case LIBXSMM_X86_INSTR_XORD_RM_R:
    case LIBXSMM_X86_INSTR_XORQ_RM_R:
    case LIBXSMM_X86_INSTR_XORB_R_RM:
    case LIBXSMM_X86_INSTR_XORW_R_RM:
    case LIBXSMM_X86_INSTR_XORD_R_RM:
    case LIBXSMM_X86_INSTR_XORQ_R_RM:
    case LIBXSMM_X86_INSTR_RDPID:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_alu_reg: Unknown instruction type: 0x%08x\n", i_alu_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_alu_instr = i_alu_instr;
    unsigned int l_gp_reg_number_src = i_gp_reg_number_src;
    unsigned int l_gp_reg_number_dest = i_gp_reg_number_dest;

    switch (i_alu_instr) {
      case LIBXSMM_X86_INSTR_ADDQ:
        l_alu_instr = LIBXSMM_X86_INSTR_ADDQ_R_RM;
        break;
      case LIBXSMM_X86_INSTR_SUBQ:
        l_alu_instr = LIBXSMM_X86_INSTR_SUBQ_R_RM;
        break;
      case LIBXSMM_X86_INSTR_MOVQ:
        l_alu_instr = LIBXSMM_X86_INSTR_MOVQ_LD;
        break;
      case LIBXSMM_X86_INSTR_CMPQ:
        l_alu_instr = LIBXSMM_X86_INSTR_CMPQ_R_RM;
        break;
      case LIBXSMM_X86_INSTR_ANDQ:
        l_alu_instr = LIBXSMM_X86_INSTR_ANDQ_R_RM;
        break;
      case LIBXSMM_X86_INSTR_POPCNT:
        l_alu_instr = LIBXSMM_X86_INSTR_POPCNTQ;
        break;
      case LIBXSMM_X86_INSTR_TZCNT:
        l_alu_instr = LIBXSMM_X86_INSTR_TZCNTQ;
        break;
    }

    /* check that we have an UNDEF for 2 src operands */
    if ( ((i_alu_instr >> 28) & 0x3) == 0x1 ) {
      if ( i_gp_reg_number_src != LIBXSMM_X86_GP_REG_UNDEF ) {
        fprintf(stderr, "libxsmm_x86_instruction_alu_reg: In case of a 1 src operand instruction (0x%08x), i_gp_reg_number_src needs to be LIBXSMM_X86_GP_REG_UNDEF!\n", i_alu_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
      l_gp_reg_number_src = 0;
    }

    /* check if we need to flip operands */
    if ( ((i_alu_instr >> 24) & 0x08 ) == 0x08 ) {
      unsigned int tmp = l_gp_reg_number_src;
      l_gp_reg_number_src  = l_gp_reg_number_dest;
      l_gp_reg_number_dest = tmp;
    }

    /* check if we have op-code extension in modrm/reg */
    if ( ((i_alu_instr >> 24) & 0x04 ) == 0x04 ) {
#if 0 /* dead condition */
      if ( ((i_alu_instr >> 28) & 0x3) == 0x1 )
#endif
      {
        l_gp_reg_number_dest = ((i_alu_instr >> 20) & 0x07);
      }
#if 0 /* dead condition */
      else {
        fprintf(stderr, "libxsmm_x86_instruction_alu_reg: In case of a op-code modrm/reg extended instruction (0x%08x) we need a single operand instruction!\n", i_alu_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
#endif
    }

    /* generate the main instruction */
    libxsmm_x86_instruction_rex_compute_2reg( io_generated_code, l_alu_instr,
            l_gp_reg_number_src, l_gp_reg_number_dest );
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
  if ( io_generated_code->code_type > 1 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_PUSHQ,
                                     LIBXSMM_X86_GP_REG_UNDEF, i_gp_reg_number );
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


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_pop_reg( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_gp_reg_number ) {
  if ( io_generated_code->code_type > 1 ) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_POPQ,
                                     LIBXSMM_X86_GP_REG_UNDEF, i_gp_reg_number );
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
      fprintf(stderr, "libxsmm_x86_instruction_mask_move: unexpected instruction number: 0x%08x\n", i_mask_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  assert((i_mask_instr & 0x300) != 0x300);
  if ( io_generated_code->code_type > 1 ) {
    /* get L bit override */
#if 0 /* see above assertion */
    const libxsmm_x86_simd_name l_vname = ( (i_mask_instr & 0x300) == 0x300) ? LIBXSMM_X86_SIMD_NAME_YMM : LIBXSMM_X86_SIMD_NAME_XMM;
#else
    const libxsmm_x86_simd_name l_vname = LIBXSMM_X86_SIMD_NAME_XMM;
#endif
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
      fprintf(stderr, "libxsmm_x86_instruction_mask_move_mem: unexpected instruction number: 0x%08x\n", i_mask_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  assert((i_mask_instr & 0x300) != 0x300);
  if ( io_generated_code->code_type > 1 ) {
    /* get L bit override */
#if 0 /* see above assertion */
    const libxsmm_x86_simd_name l_vname = ( (i_mask_instr & 0x300) == 0x300) ? LIBXSMM_X86_SIMD_NAME_YMM : LIBXSMM_X86_SIMD_NAME_XMM;
#else
    const libxsmm_x86_simd_name l_vname = LIBXSMM_X86_SIMD_NAME_XMM;
#endif
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
                                               const unsigned int      i_imm8 ) {
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
      fprintf(stderr, "libxsmm_x86_instruction_mask_compute_reg: unexpected instruction number: 0x%08x\n", i_mask_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 ) {
    /* get L bit override */
    const libxsmm_x86_simd_name l_vname = ( (i_mask_instr & 0x300) == 0x300) ? LIBXSMM_X86_SIMD_NAME_YMM : LIBXSMM_X86_SIMD_NAME_XMM;
    unsigned int l_src1 = 0;

    /* check that we have an UNDEF for 2 src operands */
    if ( ((i_mask_instr >> 28) & 3) == 2 ) {
      if ( i_mask_reg_number_src_1 != LIBXSMM_X86_VEC_REG_UNDEF ) {
        fprintf(stderr, "libxsmm_x86_instruction_mask_compute_reg: In case of a 1 src operand instruction (0x%08x), i_reg_number_src1 needs to be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_mask_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      }
      l_src1 = 0;
    } else {
      if ( i_mask_reg_number_src_1 == LIBXSMM_X86_VEC_REG_UNDEF ) {
        fprintf(stderr, "libxsmm_x86_instruction_mask_compute_reg: In case of a 2 src operand instruction (0x%08x), i_reg_number_src1 cannot be LIBXSMM_X86_VEC_REG_UNDEF!\n", i_mask_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
      } else {
        l_src1 = i_mask_reg_number_src_1;
      }
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
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
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
  /*const unsigned int i_gp_reg_idx = LIBXSMM_X86_GP_REG_UNDEF;*/
  /*const unsigned int i_scale = 1;*/

  /* TODO: check instruction set */
  LIBXSMM_UNUSED( i_instruction_set );

  if ( (i_gp_reg_base == LIBXSMM_X86_GP_REG_UNDEF) && (i_tile_config == NULL) && (i_tcontrol_instr != LIBXSMM_X86_INSTR_TILERELEASE) ) {
    fprintf(stderr, "invalid tile control!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
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

  /* TODO: add checks in debug mode */
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
        /* here we always it the gpr */
        l_third = 0x1;
        break;
      case LIBXSMM_X86_INSTR_TILERELEASE:
        l_fifth = 0xc0;
        l_gp8 = 0;
        l_regbas0 = 0;
        break;
      default:
        fprintf(stderr, "Unknown instruction in libxsmm_x86_instruction_tile_control. This is bad\n");
        break;
    }
#if 0
    if ( (i_gp_reg_idx != LIBXSMM_X86_GP_REG_UNDEF) && ((i_gp_reg_idx < LIBXSMM_X86_GP_REG_RAX) || (i_gp_reg_idx > LIBXSMM_X86_GP_REG_R15)) )
    {
       fprintf(stderr, "libxsmm_x86_instruction_tile_control is using a bogus i_gp_reg_idx\n");
       LIBXSMM_EXIT_ERROR(io_generated_code);
       return;
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
#if 0
       if ( i_gp_reg_idx == LIBXSMM_X86_GP_REG_UNDEF )
       {
#endif
          buf[i++] = (unsigned char)(0xc4);
          buf[i++] = (unsigned char)(0xe2 - l_gp8 * 0x20);
          buf[i++] = (unsigned char)(0x78 + l_third);
          buf[i++] = (unsigned char)(0x49);
          l_place = i - 1;
          buf[i++] = (unsigned char)(0x00 + l_regbas0 + l_fifth);
          if ( l_regbas0 == 4 ) buf[i++] = (unsigned char)(0x24);
#if 0
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
#endif

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
            /* TODO: handle error */
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
          /* TODO: handle error */
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
  /* TODO: check instruction set */
  LIBXSMM_UNUSED( i_instruction_set );

  /* check if passed in a correct instruction */
  switch ( i_tmove_instr ) {
    case LIBXSMM_X86_INSTR_TILELOADD:
    case LIBXSMM_X86_INSTR_TILELOADDT1:
    case LIBXSMM_X86_INSTR_TILESTORED:
    case LIBXSMM_X86_INSTR_TILEZERO:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_tile_move: unexpected instruction number: 0x%08x\n", i_tmove_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( (io_generated_code->code_type > 1) &&
       ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) ) {
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
        fprintf(stderr, "libxsmm_x86_instruction_tile_move: instruction 0x%08x requires SIB addressing\n", i_tmove_instr);
        LIBXSMM_EXIT_ERROR(io_generated_code);
        return;
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
      case LIBXSMM_X86_INSTR_TILELOADDT1: {
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
          /* TODO: handle error */
        }
      } break;
      case LIBXSMM_X86_INSTR_TILESTORED: {
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
          /* TODO: handle error */
        }
      } break;
      case LIBXSMM_X86_INSTR_TILEZERO: {
        if ( io_generated_code->code_type == 0 ) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       \"%s %%%%tmm%u\\n\\t\"\n",
                                                       l_instr_name, i_tile_reg_number );
        } else {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%tmm%u\n",
                                                       l_instr_name, i_tile_reg_number );
        }
        libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      } break;
      default:
        assert(0/*should not happen*/);
    }
  } else {
    /* general encoder error */
    fprintf(stderr, "libxsmm_x86_instruction_vec_mask_move: GENERAL ERROR\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_tile_compute( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_instruction_set,
                                           const unsigned int      i_tcompute_instr,
                                           const unsigned int      i_tile_src_reg_number_0,
                                           const unsigned int      i_tile_src_reg_number_1,
                                           const unsigned int      i_tile_dst_reg_number ) {
  /* TODO: check instruction set */
  LIBXSMM_UNUSED( i_instruction_set );

  /* check if passed in a correct instruction */
  switch ( i_tcompute_instr ) {
    case LIBXSMM_X86_INSTR_TDPBSSD:
    case LIBXSMM_X86_INSTR_TDPBSUD:
    case LIBXSMM_X86_INSTR_TDPBUSD:
    case LIBXSMM_X86_INSTR_TDPBUUD:
    case LIBXSMM_X86_INSTR_TDPBF16PS:
    case LIBXSMM_X86_INSTR_TDPFP16PS:
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_tile_compute: unexpected instruction number: 0x%08x\n", i_tcompute_instr);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( (io_generated_code->code_type > 1) &&
       ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR) && (io_generated_code->arch < LIBXSMM_X86_ALLFEAT)) ) {
    /* check if we have enough code buffer space left */
    if ( (io_generated_code->buffer_size - io_generated_code->code_size) < 20 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* invoke VEX encoder */
#if 0 /* dead condition */
    if ( ((i_tcompute_instr >> 28) & 0x3) == 3 )
#endif
    {
      libxsmm_x86_instruction_vex_compute_3reg ( io_generated_code, i_tcompute_instr, LIBXSMM_X86_SIMD_NAME_XMM,
            i_tile_src_reg_number_1, i_tile_src_reg_number_0, i_tile_dst_reg_number );
    }
#if 0 /* dead condition */
    else {
      fprintf(stderr, "libxsmm_x86_instruction_tile_compute: every instruction needs to have 3 operands\n");
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }
#endif
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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

void libxsmm_x86_instruction_register_jump_back_label( libxsmm_generated_code*     io_generated_code,
                                                  libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  /* check if we still have label we can jump to */
  if ( io_loop_label_tracker->label_count == 512 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  /* TODO: add checks in debug mode */
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

  /* TODO: add checks in debug mode */
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
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
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

  /* TODO: add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_ref = 0;
    libxsmm_jump_source l_source = io_jump_label_tracker->label_source[i_label_no];
    /* first added label to tracker */
    io_jump_label_tracker->label_address[i_label_no] = io_generated_code->code_size;
    /* patching all previous references */
    for ( l_ref = 0; l_ref < l_source.ref_count; ++l_ref ) {
      unsigned int l_jmp_instr = l_source.instr_type[l_ref];
      unsigned int l_position =   l_source.instr_addr[l_ref];
      /* This routine just does everything related to jumping. In this case, we know the destination/target */
      internal_x86_jumping ( io_generated_code, l_position, io_generated_code->code_size, l_jmp_instr );
      /* We do not need to forward the bytes here */
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

  /* TODO: add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    int l_dest_addr;
    int l_tmp;

    if ( io_jump_label_tracker->label_address[i_label_no] == 0 ) {
      l_dest_addr = -1; /* It's a forward jump to a location we have not set yet. We will assume 5-6 bytes */
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
  unsigned char vlen_encoding = 0;

  switch ( i_vector_name ) {
    case 'x':
      number_of_bytes_to_load = 16;
      vlen_encoding = 0x08;
      break;
    case 'y':
      number_of_bytes_to_load = 32;
      vlen_encoding = 0x28;
      break;
    case 'z':
      number_of_bytes_to_load = 64;
      vlen_encoding = 0x48;
      break;
    default:
      fprintf(stderr, "libxsmm_x86_instruction_full_vec_load_of_constants: strange input for i_vector_name: %c\n",i_vector_name);
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
  }

  if ( io_generated_code->code_type > 1 )
  {
    unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
    const unsigned char *cval = (const unsigned char *) &i_data[0];
    int i = io_generated_code->code_size;
    unsigned int l_maxsize = io_generated_code->buffer_size;
    int j = 0;
    int l_stop = 0;
    int l_regsize_adjustment = 0;
    int l_last_load_location = 0;
    int jmpval = 0;
    int vecval = 0;

    /* TODO: fix max. size error */
    if ( l_maxsize - i < 139 ) {
      fprintf(stderr, "libxsmm_x86_instruction_full_vec_load_of_constants: Most constant jumps need at most 139 bytes\n");
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
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
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
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
    if (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL128_SKX) {
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
      buf[i+3] = vlen_encoding;
      i += 4;
    } else {
      if ( io_generated_code->arch >= LIBXSMM_X86_AVX ) {
        buf[i] = 0xc5;
        if ( i_vec_reg_number <= 7 ) {
          buf[i+1] = (unsigned char)(0xfc + l_regsize_adjustment);
          vecval = i_vec_reg_number;
        } else {
          buf[i+1] = (unsigned char)(0x7c + l_regsize_adjustment);
          vecval = i_vec_reg_number - 8;
        }
        i += 2;
      } else {
        if ( i_vec_reg_number <= 7 ) {
          buf[i] = 0x0f;
          vecval = i_vec_reg_number;
          i += 1;
        } else {
          buf[i] = 0x44;
          buf[i+1] = 0x0f;
          vecval = i_vec_reg_number - 8;
          i += 2;
        }
      }
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
    const unsigned char *cval = (const unsigned char *) &i_data[0];
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
                                                                                                        cval[j+0],cval[j+1],cval[j+2],cval[j+3] );
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
                                                                                                      cval[j+0],cval[j+1],cval[j+2],cval[j+3] );
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
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream_gemm( libxsmm_generated_code*       io_generated_code,
                                               const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                               const unsigned int            skip_callee_save,
                                               unsigned int                  i_prefetch) {
  /* TODO: add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* TODO: this is currently System V AMD64 RTL(C) ABI only */
    unsigned char* l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (NULL == l_code_buffer || l_max_size < (l_code_size + 9)) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
    }

    if ( skip_callee_save == 0 ) {
      /* on windows we also have to save xmm6-xmm15 */
#if defined(_WIN32) || defined(__CYGWIN__)
      unsigned int l_i;
      unsigned int l_simd_store_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_ST
                                                                                    : LIBXSMM_X86_INSTR_VMOVUPS_ST;
      /* decrease rsp by 160 (10x16) */
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, 160);
      /* save 10 xmm onto the stack */
      for (l_i = 0; l_i < 10; ++l_i) {
        libxsmm_x86_instruction_vec_compute_mem_1reg_mask(io_generated_code,
          l_simd_store_instr, 'x',
          LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 144 - (l_i * 16), 0, 6 + l_i, 0, 0);
      }
      /* update code length */
      l_code_size = io_generated_code->code_size;
#endif
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
#if defined(_WIN32) || defined(__CYGWIN__)
      /* push rdi */
      l_code_buffer[l_code_size++] = 0x57;
      /* push rsi */
      l_code_buffer[l_code_size++] = 0x56;
#endif
      /* update code length */
      io_generated_code->code_size = l_code_size;
    }
  } else if ( io_generated_code->code_type == 1 ) {
    /* TODO: this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    if ( skip_callee_save == 0 ) {
      /* on windows we also have to save xmm6-xmm15 */
#if defined(_WIN32) || defined(__CYGWIN__)
      {
        unsigned int l_i;
        unsigned int l_simd_store_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_LD
                                                                                      : LIBXSMM_X86_INSTR_VMOVUPS_LD;
        char l_gp_reg_base_name[4];
        char l_instr_name[16];

        libxsmm_get_x86_gp_reg_name(LIBXSMM_X86_GP_REG_RSP, l_gp_reg_base_name, 3);

        /* decrease rsp by 160 (10x16) */
        libxsmm_get_x86_instr_name(LIBXSMM_X86_INSTR_SUBQ, l_instr_name, 15);
        l_code_length = LIBXSMM_SNPRINTF(
          l_new_code, l_max_code_length, "                       %s $%i, %%%s\n", l_instr_name, 160, l_gp_reg_base_name);
        libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length);

        libxsmm_get_x86_instr_name(l_simd_store_instr, l_instr_name, 15);
        /* save 10 xmm onto the stack */
        for (l_i = 0; l_i < 10; ++l_i) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %i(%%%s)\\n\\t\"\n",
            l_instr_name, 'x', 6 + l_i, 144 - (l_i * 16), l_gp_reg_base_name);
          libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length);
        }
      }
#endif
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
#if defined(_WIN32) || defined(__CYGWIN__)
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%rdi\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%rsi\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#endif
    }

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
void libxsmm_x86_instruction_close_stream_gemm( libxsmm_generated_code*       io_generated_code,
                                                const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                                const unsigned int            skip_callee_save,
                                                unsigned int                  i_prefetch) {
  /* TODO: add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* TODO: this is a very simple System V ABI 64 interface */
    unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (l_max_size < (l_code_size + 10)) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    if ( skip_callee_save == 0 ) {
      /* pop callee save registers */
#if defined(_WIN32) || defined(__CYGWIN__)
      /* pop rsi */
      l_code_buffer[l_code_size++] = 0x5e;
      /* pop rdi */
      l_code_buffer[l_code_size++] = 0x5f;
#endif
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

      /* on windows we also have to restore xmm6-xmm15 */
#if defined(_WIN32) || defined(__CYGWIN__)
      {
        unsigned int l_i;
        unsigned int l_simd_load_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_LD
                                                                                     : LIBXSMM_X86_INSTR_VMOVUPS_LD;
        /* update code length */
        io_generated_code->code_size = l_code_size;
        /* save 10 xmm onto the stack */
        for (l_i = 0; l_i < 10; ++l_i) {
          libxsmm_x86_instruction_vec_compute_mem_1reg_mask(io_generated_code, l_simd_load_instr, 'x', LIBXSMM_X86_GP_REG_RSP,
            LIBXSMM_X86_GP_REG_UNDEF, 0, 144 - (l_i * 16), 0, 6 + l_i, 0, 0);
        }
        /* increase rsp by 160 (10x16) */
        libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RSP, 160);
        /* update code length */
        l_code_size = io_generated_code->code_size;
      }
#endif
    }

    /* retq */
    /* TODO: I do not know if this is the correct placement in the generation process */
    l_code_buffer[l_code_size++] = 0xc3;

    /* update code length */
    io_generated_code->code_size = l_code_size;
  } else if ( io_generated_code->code_type == 1 ) {
    /* TODO: this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    if ( skip_callee_save == 0 ) {
#if defined(_WIN32) || defined(__CYGWIN__)
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%rsi\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%rdi\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#endif
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

      /* on windows we also have to restore xmm6-xmm15 */
#if defined(_WIN32) || defined(__CYGWIN__)
      {
        unsigned int l_i;
        unsigned int l_simd_load_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_LD
                                                                                     : LIBXSMM_X86_INSTR_VMOVUPS_LD;
        char l_gp_reg_base_name[4];
        char l_instr_name[16];

        libxsmm_get_x86_gp_reg_name(LIBXSMM_X86_GP_REG_RSP, l_gp_reg_base_name, 3);
        libxsmm_get_x86_instr_name(l_simd_load_instr, l_instr_name, 15);

        /* save 10 xmm onto the stack */
        for (l_i = 0; l_i < 10; ++l_i) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length,
            "                       %s %i(%%%s), %%%cmm%u\\n\\t\"\n", l_instr_name, 144 - (l_i * 16), l_gp_reg_base_name,
            'x', 6 + l_i);
          libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length);
        }
        /* increase rsp by 160 (10x16) */
        libxsmm_get_x86_instr_name(LIBXSMM_X86_INSTR_ADDQ, l_instr_name, 15);
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s $%i, %%%s\n", l_instr_name,
          160, l_gp_reg_base_name);
        libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length);
      }
#endif
    }

    /* TODO: I do not know if this is the correct placement in the generation process */
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
      if ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(B_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_b, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(B_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else if ( i_prefetch == LIBXSMM_GEMM_PREFETCH_AL2 ) {
      if ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else if ( i_prefetch == LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C ) {
      if ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch), \"m\"(B_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_pre_b, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C), \"m\"(A_prefetch), \"m\"(B_prefetch) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    } else {
      if ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      } else {
        l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(A), \"m\"(B), \"m\"(C) : \"k1\",\"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
      }
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_lea_data( libxsmm_generated_code*     io_generated_code,
                                       unsigned int                i_reg,
                                       unsigned int                i_off,
                                       libxsmm_const_data_tracker* io_const_data ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned char* l_buf = (unsigned char*) io_generated_code->generated_code;
    unsigned int l_cs = io_generated_code->code_size;

    /* Ensure we have enough space */
    if ( io_generated_code->buffer_size + 7 < l_cs ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* Ensure we have space in the fixup buffer */
    if ( 128 == io_const_data->const_data_nload_insns ) {
      fprintf( stderr, "libxsmm_x86_instruction_lea_data out of fixup space!\n" );
      LIBXSMM_EXIT_ERROR(io_generated_code);
      return;
    }

    /* lea i_reg, [rip + i_off + <FIXUP>] */
    l_buf[l_cs++] = (i_reg >= 8) ? 0x4c : 0x48;
    l_buf[l_cs++] = 0x8d;
    l_buf[l_cs++] = 0x5 + (i_reg % 8)*8;

    /* Stash the offset */
    memcpy( l_buf + l_cs, &i_off, sizeof(i_off) );

    io_const_data->const_data_pc_load_insns[io_const_data->const_data_nload_insns++] = io_generated_code->code_size;
    io_generated_code->code_size += 7;
  } else {
    fprintf(stderr, "libxsmm_x86_instruction_lea_data: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return;
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_x86_instruction_add_data( libxsmm_generated_code*     io_generated_code,
                                               const unsigned char*        i_data,
                                               unsigned int                i_ndata_bytes,
                                               unsigned int                i_alignment,
                                               unsigned int                i_append_only,
                                               libxsmm_const_data_tracker* io_const_data ) {
  i_alignment = LIBXSMM_MAX( i_alignment, 1 );

  if ( io_generated_code->code_type > 1 ) {
    unsigned char* l_data = (unsigned char*) io_const_data->const_data;
    unsigned int l_dsize = io_const_data->const_data_size;
    unsigned int l_doff, l_npad;

    /* See if we already have the data */
    if ( !i_append_only ) {
      for ( l_doff = 0; l_doff < l_dsize; l_doff += i_alignment ) {
        if ( i_ndata_bytes <= l_dsize - l_doff && !memcmp( l_data + l_doff, i_data, i_ndata_bytes) ) {
          return l_doff;
        }
      }
    }

    /* Determine how much padding is needed */
    l_npad = LIBXSMM_UP( l_dsize, i_alignment) - l_dsize;

    /* Ensure we have enough space */
    if ( l_dsize + l_npad + i_ndata_bytes > sizeof(io_const_data->const_data) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return ~0U;
    }

    /* Copy the data */
    memcpy( l_data + l_dsize + l_npad, i_data, i_ndata_bytes );

    /* Update the size */
    io_const_data->const_data_size += l_npad + i_ndata_bytes;

    /* Return the offset of the new data in the buffer */
    return l_dsize + l_npad;
  } else {
    fprintf(stderr, "libxsmm_x86_instruction_add_data: inline/pure assembly print is not supported!\n");
    LIBXSMM_EXIT_ERROR(io_generated_code);
    return 0;
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_data( libxsmm_generated_code*     io_generated_code,
                                         libxsmm_const_data_tracker* io_const_data ) {
  unsigned int l_i;
  unsigned char* l_code_buffer = (unsigned char*) io_generated_code->generated_code;
  unsigned int l_code_size = io_generated_code->code_size;
  unsigned int l_data_size = io_const_data->const_data_size;
  unsigned int l_max_size = io_generated_code->buffer_size;

  /* Handle any constant data */
  if ( l_data_size > 0 ) {
    /* Round up to a page boundary */
    l_code_size = LIBXSMM_UP( l_code_size, LIBXSMM_PAGE_MINSIZE );

    /* Ensure we have space in the code stream */
    if ( l_max_size < l_data_size + l_code_size ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL );
      return;
    }

    /* Copy the data into the buffer */
    memcpy( l_code_buffer + l_code_size, io_const_data->const_data, l_data_size );

    /* Update the data size including unused space (page-size alignment */
    io_generated_code->data_size = l_code_size + l_data_size - io_generated_code->code_size;

    /* Fill in the load address */
    for ( l_i = 0; l_i < io_const_data->const_data_nload_insns; l_i++ ) {
      unsigned int l_lea_off = io_const_data->const_data_pc_load_insns[l_i];
      unsigned int l_off, l_rip_off;

      /* Read the user-provided offset */
      memcpy( &l_off, l_code_buffer + l_lea_off + 3, sizeof(l_off) );

      l_rip_off = l_code_size - l_lea_off - 7 + l_off;
      memcpy( l_code_buffer + l_lea_off + 3, &l_rip_off, sizeof(l_rip_off) );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream_alt( libxsmm_generated_code* io_generated_code,
                                             const unsigned int      i_gp_struct_params,
                                             const unsigned int      skip_callee_save ) {
  /* TODO: add checks in debug mode */
  if ( io_generated_code->code_type > 1 ) {
    /* TODO: this is currently System V AMD64 RTL(C) ABI only */
    unsigned char* l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (NULL == l_code_buffer || l_max_size < (l_code_size + 40)) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
    }

    if ( skip_callee_save == 0 ) {
      /* on windows we also have to save xmm6-xmm15 */
#if defined(_WIN32) || defined(__CYGWIN__)
      unsigned int l_i;
      unsigned int l_simd_store_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_ST
                                                                                    : LIBXSMM_X86_INSTR_VMOVUPS_ST;
      /* decrease rsp by 160 (10x16) */
      libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, 160);
      /* save 10 xmm onto the stack */
      for (l_i = 0; l_i < 10; ++l_i) {
        libxsmm_x86_instruction_vec_compute_mem_1reg_mask(io_generated_code, l_simd_store_instr, 'x', LIBXSMM_X86_GP_REG_RSP,
          LIBXSMM_X86_GP_REG_UNDEF, 0, 144 - (l_i * 16), 0, 6 + l_i, 0, 0);
      }
      /* update code length */
      l_code_size = io_generated_code->code_size;
#endif
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
#if defined(_WIN32) || defined(__CYGWIN__)
      /* push rdi */
      l_code_buffer[l_code_size++] = 0x57;
      /* push rsi */
      l_code_buffer[l_code_size++] = 0x56;
#endif
    }

    /* update code length */
    io_generated_code->code_size = l_code_size;
  } else if ( io_generated_code->code_type == 1 ) {
    /* TODO: this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    if ( skip_callee_save == 0 ) {
      /* on windows we also have to save xmm6-xmm15 */
#if defined(_WIN32) || defined(__CYGWIN__)
      {
        unsigned int l_i;
        unsigned int l_simd_store_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_LD
                                                                                      : LIBXSMM_X86_INSTR_VMOVUPS_LD;
        char l_gp_reg_base_name[4];
        char l_instr_name[16];

        libxsmm_get_x86_gp_reg_name(LIBXSMM_X86_GP_REG_RSP, l_gp_reg_base_name, 3);

        /* decrease rsp by 160 (10x16) */
        libxsmm_get_x86_instr_name(LIBXSMM_X86_INSTR_SUBQ, l_instr_name, 15);
        l_code_length = LIBXSMM_SNPRINTF(
          l_new_code, l_max_code_length, "                       %s $%i, %%%s\n", l_instr_name, 160, l_gp_reg_base_name);
        libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length);

        libxsmm_get_x86_instr_name(l_simd_store_instr, l_instr_name, 15);
        /* save 10 xmm onto the stack */
        for (l_i = 0; l_i < 10; ++l_i) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %%%cmm%u, %i(%%%s)\\n\\t\"\n",
            l_instr_name, 'x', 6 + l_i, 144 - (l_i * 16), l_gp_reg_base_name);
          libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length);
        }
      }
#endif
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
#if defined(_WIN32) || defined(__CYGWIN__)
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%rdi\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       pushq %%rsi\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#endif
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
void libxsmm_x86_instruction_close_stream_alt( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      skip_callee_save ) {
  if ( io_generated_code->code_type > 1 ) {
    /* TODO: this is a very simple System V ABI 64 interface */
    unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
    unsigned int l_code_size = io_generated_code->code_size;
    unsigned int l_max_size = io_generated_code->buffer_size;

    if (NULL == l_code_buffer || l_max_size < (l_code_size + 10)) {
      LIBXSMM_HANDLE_ERROR(io_generated_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
      return;
    }

    if ( skip_callee_save == 0 ) {
      /* pop callee save registers */
#if defined(_WIN32) || defined(__CYGWIN__)
      /* pop rsi */
      l_code_buffer[l_code_size++] = 0x5e;
      /* pop rdi */
      l_code_buffer[l_code_size++] = 0x5f;
#endif
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

      /* on windows we also have to restore xmm6-xmm15 */
#if defined(_WIN32) || defined(__CYGWIN__)
      {
        unsigned int l_i;
        unsigned int l_simd_load_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_LD
                                                                                     : LIBXSMM_X86_INSTR_VMOVUPS_LD;
        /* update code length */
        io_generated_code->code_size = l_code_size;
        /* save 10 xmm onto the stack */
        for (l_i = 0; l_i < 10; ++l_i) {
          libxsmm_x86_instruction_vec_compute_mem_1reg_mask(io_generated_code, l_simd_load_instr, 'x', LIBXSMM_X86_GP_REG_RSP,
            LIBXSMM_X86_GP_REG_UNDEF, 0, 144 - (l_i * 16), 0, 6 + l_i, 0, 0);
        }
        /* increase rsp by 160 (10x16) */
        libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RSP, 160);
        /* update code length */
        l_code_size = io_generated_code->code_size;
      }
#endif
    }

    /* retq */
    /* TODO: I do not know if this is the correct placement in the generation process */
    l_code_buffer[l_code_size++] = 0xc3;

    /* update code length */
    io_generated_code->code_size = l_code_size;
  } else if ( io_generated_code->code_type == 1 ) {
    /* TODO: this is currently System V AMD64 RTL(C) ABI only */
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    if ( skip_callee_save  == 0 ) {
#if defined(_WIN32) || defined(__CYGWIN__)
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%rsi\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       popq %%rdi\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
#endif
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
      /* on windows we also have to restore xmm6-xmm15 */
#if defined(_WIN32) || defined(__CYGWIN__)
      {
        unsigned int l_i;
        unsigned int l_simd_load_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_LD
                                                                                     : LIBXSMM_X86_INSTR_VMOVUPS_LD;
        char l_gp_reg_base_name[4];
        char l_instr_name[16];

        libxsmm_get_x86_gp_reg_name(LIBXSMM_X86_GP_REG_RSP, l_gp_reg_base_name, 3);
        libxsmm_get_x86_instr_name(l_simd_load_instr, l_instr_name, 15);

        /* save 10 xmm onto the stack */
        for (l_i = 0; l_i < 10; ++l_i) {
          l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "                       %s %i(%%%s), %%%cmm%u\\n\\t\"\n",
            l_instr_name, 144 - (l_i * 16), l_gp_reg_base_name, 'x', 6 + l_i);
          libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length);
        }
        /* increase rsp by 160 (10x16) */
        libxsmm_get_x86_instr_name(LIBXSMM_X86_INSTR_ADDQ, l_instr_name, 15);
        l_code_length = LIBXSMM_SNPRINTF(
          l_new_code, l_max_code_length, "                       %s $%i, %%%s\n", l_instr_name, 160, l_gp_reg_base_name);
        libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length);
      }
#endif
    }

    /* TODO: I do not know if this is the correct placement in the generation process */
    l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       retq\n" );
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  } else {
    char l_new_code[1024];
    int l_max_code_length = 1023;
    int l_code_length = 0;

    if (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX ) {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(ptr) : \"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n");
    } else {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "                       : : \"m\"(ptr) : \"rax\",\"rbx\",\"rcx\",\"rdx\",\"rdi\",\"rsi\",\"r8\",\"r9\",\"r10\",\"r11\",\"r12\",\"r13\",\"r14\",\"r15\",\"zmm0\",\"zmm1\",\"zmm2\",\"zmm3\",\"zmm4\",\"zmm5\",\"zmm6\",\"zmm7\",\"zmm8\",\"zmm9\",\"zmm10\",\"zmm11\",\"zmm12\",\"zmm13\",\"zmm14\",\"zmm15\",\"zmm16\",\"zmm17\",\"zmm18\",\"zmm19\",\"zmm20\",\"zmm21\",\"zmm22\",\"zmm23\",\"zmm24\",\"zmm25\",\"zmm26\",\"zmm27\",\"zmm28\",\"zmm29\",\"zmm30\",\"zmm31\");\n");
    }
    libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
  }
}
