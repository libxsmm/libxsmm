/******************************************************************************
* Copyright (c) 2021, Friedrich Schiller University Jena                      *
* Copyright (c) 2024, IBM Corporation                                         *
* - All rights reserved.                                                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Will Trojak (IBM Corp.)
******************************************************************************/

#include "generator_ppc64le_instructions.h"

/*
 * Source:
 *   Ppc64le ISA
 *   Version 3.1
 */

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_b_conditional( unsigned int  i_instr,
                                                        unsigned char i_bo,
                                                        unsigned char i_bi,
                                                        unsigned int  i_bd ) {
  unsigned int l_instr = i_instr;

  /* set BO */
  l_instr |= (unsigned int)( (0x1f & i_bo) << (31- 6-4) );
  /* set BI */
  l_instr |= (unsigned int)( (0x1f & i_bi) << (31-11-4) );
  /* set BD */
  l_instr |= (unsigned int)( (0x3fff & i_bd) << (31-16-13) );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_fip_storage_access( unsigned int  i_instr,
                                                             unsigned char i_rs,
                                                             unsigned char i_ra,
                                                             unsigned int  i_d ) {
  unsigned int l_instr = i_instr;

  /* set RS */
  l_instr |= (unsigned int)( (0x1f & i_rs) << (31- 6-4) );
  /* set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31-11-4) );
  /* set D */
  l_instr |= (unsigned int)( (0xffff & i_d) << (31-16-15) );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_fip_arithmetic( unsigned int  i_instr,
                                                         unsigned char i_rt,
                                                         unsigned char i_ra,
                                                         unsigned int  i_si ) {
  unsigned int l_instr = i_instr;

  /* set RT */
  l_instr |= (unsigned int)( (0x1f & i_rt) << (31- 6-4) );
  /* set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31-11-4) );
  /* set SI */
  l_instr |= (unsigned int)( (0xffff & i_si) << (31-16-15) );

  return l_instr;
}

unsigned int libxsmm_ppc64le_instruction_fip_compare( unsigned int  i_instr,
                                                      unsigned char i_bf,
                                                      unsigned char i_l,
                                                      unsigned char i_ra,
                                                      unsigned int  i_si ) {
   unsigned int l_instr = i_instr;

   /* set BF */
  l_instr |= (unsigned int)( (0x7 & i_bf) << (31- 6-2) );
  /* set L */
  l_instr |= (unsigned int)( (0x1 & i_l) << (31- 10) );
  /* set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31- 11-4) );
  /* set SI */
  l_instr |= (unsigned int)( (0xffff & i_si) << (31-16-15) );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_fip_logical( unsigned int  i_instr,
                                                      unsigned char i_ra,
                                                      unsigned char i_rs,
                                                      unsigned int  i_ui ) {
  unsigned int l_instr = i_instr;

  /* set RS */
  l_instr |= (unsigned int)( (0x1f & i_rs) << (31- 6-4) );
  /* set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31-11-4) );
  /* set UI */
  l_instr |= (unsigned int)( (0xffff & i_ui) << (31-16-15) );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_fip_rotate( unsigned int  i_instr,
                                                     unsigned char i_ra,
                                                     unsigned char i_rs,
                                                     unsigned int  i_sh,
                                                     unsigned int  i_mb ) {
  unsigned int l_instr = i_instr;

  /* set RS */
  l_instr |= (unsigned int)( (0x1f & i_rs) << (31- 6-4) );
  /* set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31-11-4) );
  /* set first 5 bits of sh */
  l_instr |= (unsigned int)( (0x1f & i_sh) << (31-16-4) );
  /* set mb */
  l_instr |= (unsigned int)( (0x3f & i_mb) << (31-21-5) );

  /* set last bit of sh */
  l_instr |= (unsigned int)( (0x20 & i_sh) >> 4 );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_fip_system( unsigned int  i_instr,
                                                     unsigned char i_rs,
                                                     unsigned int  i_spr ) {
  unsigned int l_instr = i_instr;

  /* set RS */
  l_instr |= (unsigned int)( (0x1f & i_rs)   << (31- 6-4) );
  /* set SPR */
  l_instr |= (unsigned int)( (0x3ff & i_spr) << (31-11-9) );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_flp_storage_access( unsigned int  i_instr,
                                                             unsigned char i_frs,
                                                             unsigned char i_ra,
                                                             unsigned int  i_d ) {
  unsigned int l_instr = i_instr;

  /* set FRS */
  l_instr |= (unsigned int)( (0x1f & i_frs) << (31- 6-4) );
  /* set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31-11-4) );
  /* set D */
  l_instr |= (unsigned int)( (0xffff & i_d) << (31-16-15) );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_vec_storage_access( unsigned int  i_instr,
                                                             unsigned char i_vrt,
                                                             unsigned char i_ra,
                                                             unsigned char i_rb ) {
  unsigned int l_instr = i_instr;

  /* set VRT */
  l_instr |= (unsigned int)( (0x1f & i_vrt) << (31- 6-4) );
  /* set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31-11-4) );
  /* set RB */
  l_instr |= (unsigned int)( (0x1f & i_rb) << (31-16-4) );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_vsx_storage_access( unsigned int  i_instr,
                                                             unsigned char i_xt,
                                                             unsigned char i_ra,
                                                             unsigned char i_rb ) {
  unsigned int l_instr = i_instr;

  /* set T */
  l_instr |= (unsigned int)( (0x1f & i_xt) << (31- 6-4) );
  /* set RA */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31-11-4) );
  /* set RB */
  l_instr |= (unsigned int)( (0x1f & i_rb) << (31-16-4) );

  /* set TX */
  l_instr |= (unsigned int)( (0x20 & i_xt) >> 5 );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_vsx_vector_bfp_madd( unsigned int  i_instr,
                                                              unsigned char i_xt,
                                                              unsigned char i_xa,
                                                              unsigned char i_xb ) {
  unsigned int l_instr = i_instr;

  /* set T */
  l_instr |= (unsigned int)( (0x1f & i_xt) << (31- 6-4) );
  /* set A */
  l_instr |= (unsigned int)( (0x1f & i_xa) << (31-11-4) );
  /* set B */
  l_instr |= (unsigned int)( (0x1f & i_xb) << (31-16-4) );

  /* set AX */
  l_instr |= (unsigned int)( (0x20 & i_xa) >> 3 );
  /* set BX */
  l_instr |= (unsigned int)( (0x20 & i_xb) >> 4 );
  /* set TX */
  l_instr |= (unsigned int)( (0x20 & i_xt) >> 5 );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_vsx_vector_permute_byte_reverse( unsigned int  i_instr,
                                                                          unsigned char i_xt,
                                                                          unsigned char i_xb ) {
  unsigned int l_instr = i_instr;

  /* set T */
  l_instr |= (unsigned int)( (0x1f & i_xt) << (31- 6-4) );
  /* set B */
  l_instr |= (unsigned int)( (0x1f & i_xb) << (31-16-4) );

  /* set BX */
  l_instr |= (unsigned int)( (0x20 & i_xb) >> 4 );
  /* set TX */
  l_instr |= (unsigned int)( (0x20 & i_xt) >> 5 );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_generic_2( unsigned int i_instr,
                                                    unsigned int i_arg0,
                                                    unsigned int i_arg1 ) {
  unsigned int l_instr = 0;

  if( i_instr == LIBXSMM_PPC64LE_INSTR_MTSPR ) {
    l_instr = libxsmm_ppc64le_instruction_fip_system( i_instr,
                                                    i_arg0,
                                                    i_arg1 );
  }
  else if( i_instr == LIBXSMM_PPC64LE_INSTR_XXBRD ||
           i_instr == LIBXSMM_PPC64LE_INSTR_XXBRW ) {
    l_instr = libxsmm_ppc64le_instruction_vsx_vector_permute_byte_reverse( i_instr,
                                                                           i_arg0,
                                                                           i_arg1 );
  }
  else {
    fprintf(stderr, "libxsmm_ppc64le_instruction_generic_2: unsupported instruction!\n");
    exit(-1);
  }

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_generic_3( unsigned int i_instr,
                                                    unsigned int i_arg0,
                                                    unsigned int i_arg1,
                                                    unsigned int i_arg2 ) {
  unsigned int l_instr = 0;

  if( i_instr == LIBXSMM_PPC64LE_INSTR_BC ) {
    l_instr = libxsmm_ppc64le_instruction_b_conditional( i_instr,
                                                         i_arg0,
                                                         i_arg1,
                                                         i_arg2 );
  }
  else if( i_instr == LIBXSMM_PPC64LE_INSTR_LD ||
           i_instr == LIBXSMM_PPC64LE_INSTR_STD ) {
    l_instr = libxsmm_ppc64le_instruction_fip_storage_access( i_instr,
                                                              i_arg0,
                                                              i_arg1,
                                                              i_arg2 );
  }
  else if( i_instr == LIBXSMM_PPC64LE_INSTR_ADDI ) {
    l_instr = libxsmm_ppc64le_instruction_fip_arithmetic( i_instr,
                                                          i_arg0,
                                                          i_arg1,
                                                          i_arg2 );
  }
  else if( i_instr == LIBXSMM_PPC64LE_INSTR_ORI ) {
    l_instr = libxsmm_ppc64le_instruction_fip_logical( i_instr,
                                                       i_arg0,
                                                       i_arg1,
                                                       i_arg2 );
  }
  else if( i_instr == LIBXSMM_PPC64LE_INSTR_LFD ||
           i_instr == LIBXSMM_PPC64LE_INSTR_STFD ) {
    l_instr = libxsmm_ppc64le_instruction_flp_storage_access( i_instr,
                                                              i_arg0,
                                                              i_arg1,
                                                              i_arg2 );
  }
  else if( i_instr == LIBXSMM_PPC64LE_INSTR_LVX ||
           i_instr == LIBXSMM_PPC64LE_INSTR_STVX ) {
    l_instr = libxsmm_ppc64le_instruction_vec_storage_access( i_instr,
                                                              i_arg0,
                                                              i_arg1,
                                                              i_arg2 );
  }
  else if( i_instr == LIBXSMM_PPC64LE_INSTR_LXVW4X ||
           i_instr == LIBXSMM_PPC64LE_INSTR_STXVW4X ||
           i_instr == LIBXSMM_PPC64LE_INSTR_LXVWSX ||
           i_instr == LIBXSMM_PPC64LE_INSTR_LXVLL ||
           i_instr == LIBXSMM_PPC64LE_INSTR_STXVLL ) {
    l_instr = libxsmm_ppc64le_instruction_vsx_storage_access( i_instr,
                                                              i_arg0,
                                                              i_arg1,
                                                              i_arg2 );
  }
  else if( i_instr == LIBXSMM_PPC64LE_INSTR_XVMADDASP ) {
    l_instr = libxsmm_ppc64le_instruction_vsx_vector_bfp_madd( i_instr,
                                                               i_arg0,
                                                               i_arg1,
                                                               i_arg2 );
  }
  else {
    fprintf(stderr, "libxsmm_ppc64le_instruction_generic_3: unsupported instruction!\n");
    exit(-1);
  }

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instruction_generic_4( unsigned int i_instr,
                                                    unsigned int i_arg0,
                                                    unsigned int i_arg1,
                                                    unsigned int i_arg2,
                                                    unsigned int i_arg3 ) {
  unsigned int l_instr = 0;

  if( i_instr == LIBXSMM_PPC64LE_INSTR_CMPI ) {
    l_instr = libxsmm_ppc64le_instruction_fip_compare( i_instr,
                                                       i_arg0,
                                                       i_arg1,
                                                       i_arg2,
                                                       i_arg3 );
  }
  else if( i_instr == LIBXSMM_PPC64LE_INSTR_RLDICR ) {
    l_instr = libxsmm_ppc64le_instruction_fip_rotate( i_instr,
                                                      i_arg0,
                                                      i_arg1,
                                                      i_arg2,
                                                      i_arg3 );
  }
  else {
    fprintf(stderr, "libxsmm_ppc64le_instruction_generic_4: unsupported instruction!\n");
    exit(-1);
  }

  return l_instr;
}

LIBXSMM_API_INTERN
void libxsmm_ppc64le_instruction_2( libxsmm_generated_code * io_generated_code,
                                    unsigned int             i_instr,
                                    unsigned int             i_arg0,
                                    unsigned int             i_arg1 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int   l_code_head = io_generated_code->code_size / 4;
    unsigned int * l_code      = (unsigned int*) io_generated_code->generated_code;
    l_code[l_code_head] = libxsmm_ppc64le_instruction_generic_2( i_instr,
                                                                 i_arg0,
                                                                 i_arg1 );
    io_generated_code->code_size += 4;
  }
  else {
    fprintf(stderr, "libxsmm_ppc64le_instruction_2: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_ppc64le_instruction_3( libxsmm_generated_code * io_generated_code,
                                    unsigned int             i_instr,
                                    unsigned int             i_arg0,
                                    unsigned int             i_arg1,
                                    unsigned int             i_arg2 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int   l_code_head = io_generated_code->code_size / 4;
    unsigned int * l_code      = (unsigned int*) io_generated_code->generated_code;
    l_code[l_code_head] = libxsmm_ppc64le_instruction_generic_3( i_instr,
                                                                 i_arg0,
                                                                 i_arg1,
                                                                 i_arg2 );
    io_generated_code->code_size += 4;
  }
  else {
    fprintf(stderr, "libxsmm_ppc64le_instruction_3: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_ppc64le_instruction_4( libxsmm_generated_code * io_generated_code,
                                    unsigned int             i_instr,
                                    unsigned int             i_arg0,
                                    unsigned int             i_arg1,
                                    unsigned int             i_arg2,
                                    unsigned int             i_arg3 ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int   l_code_head = io_generated_code->code_size / 4;
    unsigned int * l_code      = (unsigned int*) io_generated_code->generated_code;
    l_code[l_code_head] = libxsmm_ppc64le_instruction_generic_4( i_instr,
                                                                 i_arg0,
                                                                 i_arg1,
                                                                 i_arg2,
                                                                 i_arg3 );
    io_generated_code->code_size += 4;
  }
  else {
    fprintf(stderr, "libxsmm_ppc64le_instruction_4: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_ppc64le_instruction_open_stream( libxsmm_generated_code * io_generated_code,
                                              unsigned short           i_gprMax,
                                              unsigned short           i_fprMax,
                                              unsigned short           i_vsrMax ) {
  /* decrease stack pointer */
  libxsmm_ppc64le_instruction_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 LIBXSMM_PPC64LE_GPR_SP,
                                 LIBXSMM_PPC64LE_GPR_SP,
                                 -512 );

  /* save general purpose registers */
  for( int l_gp = 13; l_gp <= i_gprMax; l_gp++ ) {
    libxsmm_ppc64le_instruction_3( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_STD,
                                   l_gp,
                                   LIBXSMM_PPC64LE_GPR_SP,
                                   (l_gp-13)*8 );
  }

  /* save floating point registers */
  for( int l_fl = 14; l_fl <= i_fprMax; l_fl++ ) {
    libxsmm_ppc64le_instruction_3( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_STFD,
                                   l_fl,
                                   LIBXSMM_PPC64LE_GPR_SP,
                                   (l_fl-14)*8 + 152 );
  }

  /* save vector registers */
  if( i_vsrMax >= 20 ) {
    libxsmm_ppc64le_instruction_3( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_ADDI,
                                   LIBXSMM_PPC64LE_GPR_R11,
                                   LIBXSMM_PPC64LE_GPR_SP,
                                   304 );
  }

  for( int l_ve = 20; l_ve <= i_vsrMax; l_ve++ ) {
    libxsmm_ppc64le_instruction_3( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_STVX,
                                   l_ve,
                                   0,
                                   LIBXSMM_PPC64LE_GPR_R11 );
    if( l_ve != 31 ) {
      libxsmm_ppc64le_instruction_3( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_ADDI,
                                     LIBXSMM_PPC64LE_GPR_R11,
                                     LIBXSMM_PPC64LE_GPR_R11,
                                     16 );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_ppc64le_instruction_close_stream( libxsmm_generated_code * io_generated_code,
                                               unsigned short           i_gprMax,
                                               unsigned short           i_fprMax,
                                               unsigned short           i_vsrMax ) {
  /* restore general purpose registers */
  for( int l_gp = 13; l_gp <= i_gprMax; l_gp++ ) {
    libxsmm_ppc64le_instruction_3( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_LD,
                                   l_gp,
                                   LIBXSMM_PPC64LE_GPR_SP,
                                   (l_gp-13)*8 );
  }

  /* restore floating point registers */
  for( int l_fl = 14; l_fl <= i_fprMax; l_fl++ ) {
    libxsmm_ppc64le_instruction_3( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_LFD,
                                   l_fl,
                                   LIBXSMM_PPC64LE_GPR_SP,
                                   (l_fl-14)*8 + 152 );
  }

  /* restore vector register */
  if( i_vsrMax >= 20 ) {
    libxsmm_ppc64le_instruction_3( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_ADDI,
                                   LIBXSMM_PPC64LE_GPR_R11,
                                   LIBXSMM_PPC64LE_GPR_SP,
                                   304 );
  }

  for( int l_ve = 20; l_ve <= i_vsrMax; l_ve++ ) {
    libxsmm_ppc64le_instruction_3( io_generated_code,
                                   LIBXSMM_PPC64LE_INSTR_LVX,
                                   l_ve,
                                   0,
                                   LIBXSMM_PPC64LE_GPR_R11 );
    if( l_ve != 31 ) {
      libxsmm_ppc64le_instruction_3( io_generated_code,
                                     LIBXSMM_PPC64LE_INSTR_ADDI,
                                     LIBXSMM_PPC64LE_GPR_R11,
                                     LIBXSMM_PPC64LE_GPR_R11,
                                     16 );
    }
  }

  /* increase stack pointer */
  libxsmm_ppc64le_instruction_3( io_generated_code,
                                 LIBXSMM_PPC64LE_INSTR_ADDI,
                                 1,
                                 1,
                                 512 );

  /* return statement */
  unsigned int   l_code_head = io_generated_code->code_size / 4;
  unsigned int * l_code      = (unsigned int*) io_generated_code->generated_code;
  l_code[l_code_head] = 0x4e800020;
  io_generated_code->code_size += 4;
}

LIBXSMM_API_INTERN
void libxsmm_ppc64le_instruction_register_jump_back_label( libxsmm_generated_code     * io_generated_code,
                                                           libxsmm_loop_label_tracker * io_loop_label_tracker ) {
  /* check if we still have label we can jump to */
  if ( io_loop_label_tracker->label_count == 512 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code,
                          LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    int l_lab = io_loop_label_tracker->label_count;
    io_loop_label_tracker->label_count++;
    io_loop_label_tracker->label_address[l_lab] = io_generated_code->code_size;
  }
  else {
    fprintf(stderr, "libxsmm_ppc64le_instruction_cond_jump_back_to_label: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_ppc64le_instruction_cond_jump_back_to_label( libxsmm_generated_code     * io_generated_code,
                                                          unsigned int                 i_gpr,
                                                          libxsmm_loop_label_tracker * io_loop_label_tracker ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int   l_lab = --io_loop_label_tracker->label_count;
    unsigned int   l_b_dst = (io_loop_label_tracker->label_address[l_lab])/4;
    unsigned int   l_code_head = io_generated_code->code_size/4;
    unsigned int * l_code = (unsigned int *)io_generated_code->generated_code;

    /* branch immediate */
    int l_b_imm = (int)l_b_dst - (int)l_code_head;

    /* compare GPR to 0 */
    l_code[l_code_head] = libxsmm_ppc64le_instruction_fip_compare( LIBXSMM_PPC64LE_INSTR_CMPI,
                                                                   0,
                                                                   0,
                                                                   i_gpr,
                                                                   0 );

    /* branch if equal */
    l_code[l_code_head+1] = libxsmm_ppc64le_instruction_b_conditional( LIBXSMM_PPC64LE_INSTR_BC,
                                                                       4,
                                                                       2,
                                                                       l_b_imm-1 );

    /* advance code head */
    io_generated_code->code_size += 4+4;
  }
  else {
    fprintf(stderr, "libxsmm_ppc64le_instruction_cond_jump_back_to_label: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_ppc64le_instruction_cond_jump_back_to_label_ctr( libxsmm_generated_code     * io_generated_code,
                                                              libxsmm_loop_label_tracker * io_loop_label_tracker ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int   l_lab = --io_loop_label_tracker->label_count;
    unsigned int   l_b_dst = (io_loop_label_tracker->label_address[l_lab])/4;
    unsigned int   l_code_head = io_generated_code->code_size/4;
    unsigned int * l_code = (unsigned int *)io_generated_code->generated_code;

    /* branch immediate */
    int l_b_imm = (int)l_b_dst - (int)l_code_head;

    /* bdnz */
    l_code[l_code_head] = libxsmm_ppc64le_instruction_generic_3( LIBXSMM_PPC64LE_INSTR_BC,
                                                                 16,
                                                                 0,
                                                                 l_b_imm );

    /* advance code head */
    io_generated_code->code_size += 4;
  }
  else {
    fprintf(stderr, "libxsmm_ppc64le_instruction_cond_jump_back_to_label_ctr: inline/pure assembly print is not supported!\n");
    exit(-1);
  }
}
