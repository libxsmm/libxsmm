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
 *   Power ISA
 *   Version 3.1
 */


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_b_form( unsigned int  i_instr,
                                           unsigned char i_bo,
                                           unsigned char i_bi,
                                           unsigned int  i_bd ) {
  unsigned int l_instr = i_instr;

  /* set BO */
  l_instr |= (unsigned int)( (0x1f & i_bo) << (31 - 6 - 4) );
  /* set BI */
  l_instr |= (unsigned int)( (0x1f & i_bi) << (31 - 11 - 4) );
  /* set BD */
  l_instr |= (unsigned int)( (0x3fff & i_bd) << (31 - 16 - 13) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_b_form_al( unsigned int  i_instr,
                                              unsigned char i_bo,
                                              unsigned char i_bi,
                                              unsigned int  i_bd,
                                              unsigned char i_aa,
                                              unsigned char i_lk ) {
  unsigned int l_instr = i_instr;

  /* set BO */
  l_instr |= (unsigned int)( (0x1f & i_bo) << (31 - 6 - 4) );
  /* set BI */
  l_instr |= (unsigned int)( (0x1f & i_bi) << (31 - 11 - 4) );
  /* set BD */
  l_instr |= (unsigned int)( (0x3fff & i_bd) << (31 - 16 - 13) );
  /* set AA */
  l_instr |= (unsigned int)( (0x01 & i_aa) << (31 - 30 - 0) );
  /* set LK */
  l_instr |= (unsigned int)( (0x01 & i_lk) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_d_form( unsigned int  i_instr,
                                           unsigned char i_t,
                                           unsigned char i_a,
                                           unsigned int  i_d ) {
  unsigned int l_instr = i_instr;

  /* set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* set A */
  l_instr |= (unsigned int)( (0x0f & i_a) << (31 - 11 - 4) );
  /* set D */
  l_instr |= (unsigned int)( (0xffff & i_d) << (31 - 16 - 15) );

  return l_instr;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_dq_form_x( unsigned int  i_instr,
                                              unsigned char i_t,
                                              unsigned char i_ra,
                                              unsigned int  i_dq,
                                              unsigned char i_x ) {
  unsigned int l_instr = i_instr;

  /* set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* set A */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31 - 11 - 4) );
  /* set DQ */
  l_instr |= (unsigned int)( (0x0fff & i_dq) << (31 - 16 - 11) );
  /* set X */
  l_instr |= (unsigned int)( (0x01 & i_pt) << (31 - 28 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_ds_form( unsigned int  i_instr,
                                            unsigned char i_s,
                                            unsigned char i_a,
                                            unsigned int  i_d ) {
  unsigned int l_instr = i_instr;

  /* set S */
  l_instr |= (unsigned int)( (0x1f & i_s) << (31 - 6 - 4) );
  /* set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* set D */
  l_instr |= (unsigned int)( (0x03ff & i_d) << (31 - 16 - 13) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_md_form( unsigned int  i_instr,
                                            unsigned char i_rs,
                                            unsigned char i_ra,
                                            unsigned char i_sh,
                                            unsigned char i_m,
                                            unsigned char i_sh2,
                                            unsigned char i_rc ) {
  unsigned int l_instr = i_instr;

  /* Set RS */
  l_instr |= (unsigned int)( (0x1f & i_rs) << (31 - 6 - 4) );
  /* Set RS */
  l_instr |= (unsigned int)( (0x1f & i_ra) << (31 - 11 - 4) );
  /* Set sh */
  l_instr |= (unsigned int)( (0x1f & i_sh) << (31 - 16 - 4) );
  /* Set mb/me */
  l_instr |= (unsigned int)( (0x3f & i_m) << (31 - 21 - 5) );
  /* Set sh */
  l_instr |= (unsigned int)( (0x01 & i_sh2) << (31 - 30 - 0) );
  /* Set Rc */
  l_instr |= (unsigned int)( (0x01 & i_rc) << (31 - 31 - 0) );

  return l_instr

}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_vx_form_vrb( unsigned int  i_instr,
                                                unsigned char i_vrb ) {
  unsigned int l_instr = i_instr;

  /* Set VRB */
  l_instr |= (unsigned int)( (0x1f & i_vrb) << (31 - 16 - 4) );

  return l_instr
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_vx_form_vrt( unsigned int  i_instr,
                                                unsigned char i_vrt ) {
  unsigned int l_instr = i_instr;

  /* Set VRT */
  l_instr |= (unsigned int)( (0x1f & i_vrt) << (31 - 6 - 4) );

  return l_instr
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form( unsigned int  i_instr,
                                           unsigned char i_t,
                                           unsigned char i_a,
                                           unsigned char i_b,
                                           unsigned char i_x ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set X */
  l_instr |= (unsigned int)( (0x01 & i_x) << (31 - 31 - 0) );

  return l_instr
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_3( unsigned int  i_instr,
                                             unsigned char i_a ) {
  unsigned int l_instr = i_instr;

  /* Set A */
  l_instr |= (unsigned int)( (0x07 & i_bf) << (31 - 6 - 2) );

  return l_instr
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_33( unsigned int  i_instr,
                                              unsigned char i_bf,
                                              unsigned char i_bfa ) {
  unsigned int l_instr = i_instr;

  /* Set BF */
  l_instr |= (unsigned int)( (0x07 & i_bf) << (31 - 6 - 2) );
  /* Set BFA */
  l_instr |= (unsigned int)( (0x07 & i_bfa) << (31 - 11 - 2) );

  return l_instr
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_355( unsigned int  i_instr,
                                               unsigned char i_bf,
                                               unsigned char i_a,
                                               unsigned char i_b ) {
  unsigned int l_instr = i_instr;

  /* Set BF */
  l_instr |= (unsigned int)( (0x07 & i_bf) << (31 - 6 - 2) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );

  return l_instr
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_x_form_555( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );

  return l_instr
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xfx_form( unsigned int  i_instr,
                                             unsigned char i_t,
                                             unsigned int  i_r ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x07 & i_t) << (31 - 6 - 4) );
  /* Set R */
  l_instr |= (unsigned int)( (0x03ff & i_r) << (31 - 11 - 9) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx2_form_2( unsigned int  i_instr,
                                               unsigned char i_rt,
                                               unsigned char i_b,
                                               unsigned char i_bx ) {
  unsigned int l_instr = i_instr;

  /* Set RT */
  l_instr |= (unsigned int)( (0x1f & i_rt) << (31 - 6 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx2_form_3( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_b,
                                               unsigned char i_bx,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx2_form_4( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_b,
                                               unsigned char i_uim,
                                               unsigned char i_bx,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set UIM */
  l_instr |= (unsigned int)( (0x03 & i_uim) << (31 - 14 - 1) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form( unsigned int  i_instr,
                                             unsigned char i_t,
                                             unsigned char i_a,
                                             unsigned char i_b) {
  unsigned int l_instr = i_instr;

  /* Set T */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_0( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_ax,
                                               unsigned char i_bx ) {
  unsigned int l_instr = i_instr;

  /* Set AT */
  l_instr |= (unsigned int)( (0x07 & i_t) << (31 - 6 - 2) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set AX */
  l_instr |= (unsigned int)( (0x01 & i_ax) << (31 - 29 - 0) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_3( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_w,
                                               unsigned char i_ax,
                                               unsigned char i_bx,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set AT */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set W */
  l_instr |= (unsigned int)( (0x03 & i_w) << (31 - 22 - 1) );
  /* Set AX */
  l_instr |= (unsigned int)( (0x01 & i_ax) << (31 - 29 - 0) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_5( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_rc,
                                               unsigned char i_ax,
                                               unsigned char i_bx,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set AT */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set RC */
  l_instr |= (unsigned int)( (0x01 & i_rc) << (31 - 21 - 0) );
  /* Set AX */
  l_instr |= (unsigned int)( (0x01 & i_ax) << (31 - 29 - 0) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx3_form_6( unsigned int  i_instr,
                                               unsigned char i_t,
                                               unsigned char i_a,
                                               unsigned char i_b,
                                               unsigned char i_ax,
                                               unsigned char i_bx,
                                               unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set AT */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set AX */
  l_instr |= (unsigned int)( (0x01 & i_ax) << (31 - 29 - 0) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 31 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_xx4_form( unsigned int  i_instr,
                                             unsigned char i_t,
                                             unsigned char i_a,
                                             unsigned char i_b,
                                             unsigned char i_c,
                                             unsigned char i_cx,
                                             unsigned char i_ax,
                                             unsigned char i_bx,
                                             unsigned char i_tx ) {
  unsigned int l_instr = i_instr;

  /* Set AT */
  l_instr |= (unsigned int)( (0x1f & i_t) << (31 - 6 - 4) );
  /* Set A */
  l_instr |= (unsigned int)( (0x1f & i_a) << (31 - 11 - 4) );
  /* Set B */
  l_instr |= (unsigned int)( (0x1f & i_b) << (31 - 16 - 4) );
  /* Set C */
  l_instr |= (unsigned int)( (0x1f & i_c) << (31 - 21 - 4) );
  /* Set CX */
  l_instr |= (unsigned int)( (0x01 & i_cx) << (31 - 28 - 0) );
  /* Set AX */
  l_instr |= (unsigned int)( (0x01 & i_ax) << (31 - 29 - 0) );
  /* Set BX */
  l_instr |= (unsigned int)( (0x01 & i_bx) << (31 - 30 - 0) );
  /* Set TX */
  l_instr |= (unsigned int)( (0x01 & i_tx) << (31 - 30 - 0) );

  return l_instr;
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_1_wrapper( unsigned int i_instr,
                                              unsigned int i_1 ) {
  switch( i_instr ) {
  /* VX (vrb) form */
  case LIBXSMM_PPC64LE_INSTR_MTVSCR:
    return libxsmm_ppc64le_vx_form_vrb( i_instr, (unsigned char)i_1 );
  /* VX (vrt) form */
  case LIBXSMM_PPC64LE_INSTR_MFVSCR:
    return libxsmm_ppc64le_vx_form_vrt( i_instr, (unsigned char)i_1 );
  /* X (3) form */
  case LIBXSMM_PPC64LE_INSTR_XXMFACC:
  case LIBXSMM_PPC64LE_INSTR_XXMTACC:
  case LIBXSMM_PPC64LE_INSTR_XXSETACCZ:
    return libxsmm_ppc64le_x_form_3( i_instr, (unsigned char)i_1 );
  default:
    LIBXSMM_HANDLE_ERROR( i_instr, LIBXSMM_ERR_UNKNOWN_OPERATION );
  }
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_2_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2 ) {
  switch( i_instr ) {
  /* X (33) form */
  case LIBXSMM_PPC64LE_INSTR_MCRFS:
    return libxsmm_ppc64le_x_form_33( i_instr, (unsigned char)i_1, (unsigned char)i_2 );
  /* XFX form */
  case LIBXSMM_PPC64LE_INSTR_MTSPR:
      return libxsmm_ppc64le_xfx_form( i_instr, (unsigned char)i_1, (unsigned int)i_2 );
  default:
    LIBXSMM_HANDLE_ERROR( i_instr, LIBXSMM_ERR_UNKNOWN_OPERATION );
  }
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_3_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3 ) {
  switch( i_instr ) {
  /* B form */
  case LIBXSMM_PPC64LE_INSTR_BC:
    return libxsmm_ppc64le_b_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3 );
  /* D form */
  case LIBXSMM_PPC64LE_INSTR_LFD:
  case LIBXSMM_PPC64LE_INSTR_STFD:
  case LIBXSMM_PPC64LE_INSTR_ORI:
  case LIBXSMM_PPC64LE_INSTR_ANDI:
  case LIBXSMM_PPC64LE_INSTR_ADDI:
    return libxsmm_ppc64le_d_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3 );
  /* DS form */
  case LIBXSMM_PPC64LE_INSTR_STFDP:
  case LIBXSMM_PPC64LE_INSTR_STXSD:
  case LIBXSMM_PPC64LE_INSTR_STXSSP:
  case LIBXSMM_PPC64LE_INSTR_STD:
  case LIBXSMM_PPC64LE_INSTR_STDU:
  case LIBXSMM_PPC64LE_INSTR_STQ:
  case LIBXSMM_PPC64LE_INSTR_LD:
    return libxsmm_ppc64le_ds_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3 );
  /* X (355) form */
  case LIBXSMM_PPC64LE_INSTR_FCMPU:
  case LIBXSMM_PPC64LE_INSTR_FCMPO:
  case LIBXSMM_PPC64LE_INSTR_LVEBX:
    return libxsmm_ppc64le_x_form_355( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
  /* X (555) form */
  case LIBXSMM_PPC64LE_INSTR_LVEHX:
  case LIBXSMM_PPC64LE_INSTR_LVEWX:
  case LIBXSMM_PPC64LE_INSTR_LVSL:
  case LIBXSMM_PPC64LE_INSTR_LVSR:
  case LIBXSMM_PPC64LE_INSTR_STVX:
  case LIBXSMM_PPC64LE_INSTR_STVXL:
  case LIBXSMM_PPC64LE_INSTR_STVEBX:
  case LIBXSMM_PPC64LE_INSTR_STVEHX:
  case LIBXSMM_PPC64LE_INSTR_STVEWX:
  case LIBXSMM_PPC64LE_INSTR_LVX:
  case LIBXSMM_PPC64LE_INSTR_LVXL:
    return libxsmm_ppc64le_x_form_555( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3 );
  /* XX2 (2) form */
  case LIBXSMM_PPC64LE_INSTR_XSXEXPDP:
  case LIBXSMM_PPC64LE_INSTR_XSXSIGDP:
    return libxsmm_ppc64le_xx2_form_2( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3);
  default:
    LIBXSMM_HANDLE_ERROR( i_instr, LIBXSMM_ERR_UNKNOWN_OPERATION );
  }
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_4_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4 ) {
  switch( i_instr ) {
  /* D (bf) form */
  case LIBXSMM_PPC64LE_INSTR_CMPI:
    return libxsmm_ppc64le_d_form_bf( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4 );
  /* DQ (x) form */
  case LIBXSMM_PPC64LE_INSTR_LXV:
  case LIBXSMM_PPC64LE_INSTR_STXV:
    return libxsmm_ppc64le_instr_dq_form_x( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4 );
  /* X form */
  case LIBXSMM_PPC64LE_INSTR_LXVW4X:
  case LIBXSMM_PPC64LE_INSTR_LXVWSX:
  case LIBXSMM_PPC64LE_INSTR_STXVW4X:
  case LIBXSMM_PPC64LE_INSTR_LXVLL:
  case LIBXSMM_PPC64LE_INSTR_STXVLL:
  case LIBXSMM_PPC64LE_INSTR_XXGENPCVBM:
  case LIBXSMM_PPC64LE_INSTR_XXGENPCVWM:
  case LIBXSMM_PPC64LE_INSTR_XXGENPCVHM:
  case LIBXSMM_PPC64LE_INSTR_XXGENPCVDM:
  case LIBXSMM_PPC64LE_INSTR_XSIEXPDP:
    return libxsmm_ppc64le_x_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4 );
  /* XX2 (3) form */
  case LIBXSMM_PPC64LE_INSTR_XSCVDPUXWS:
  case LIBXSMM_PPC64LE_INSTR_XSCVDPSXWS:
  case LIBXSMM_PPC64LE_INSTR_XVCVSPUXWS:
  case LIBXSMM_PPC64LE_INSTR_XVCVSPSXWS:
  case LIBXSMM_PPC64LE_INSTR_XVCVUXWSP:
  case LIBXSMM_PPC64LE_INSTR_XVCVSXWSP:
  case LIBXSMM_PPC64LE_INSTR_XVCVDPUXWS:
  case LIBXSMM_PPC64LE_INSTR_XVCVDPSXWS:
  case LIBXSMM_PPC64LE_INSTR_XVCVUXWDP:
  case LIBXSMM_PPC64LE_INSTR_XVCVSXWDP:
  case LIBXSMM_PPC64LE_INSTR_XSCVUXDSP:
  case LIBXSMM_PPC64LE_INSTR_XSCVSXDSP:
  case LIBXSMM_PPC64LE_INSTR_XSCVDPUXDS:
  case LIBXSMM_PPC64LE_INSTR_XSCVDPSXDS:
  case LIBXSMM_PPC64LE_INSTR_XSCVUXDDP:
  case LIBXSMM_PPC64LE_INSTR_XSCVSXDDP:
  case LIBXSMM_PPC64LE_INSTR_XVCVSPUXDS:
  case LIBXSMM_PPC64LE_INSTR_XVCVSPSXDS:
  case LIBXSMM_PPC64LE_INSTR_XVCVUXDSP:
  case LIBXSMM_PPC64LE_INSTR_XVCVSXDSP:
  case LIBXSMM_PPC64LE_INSTR_XVCVDPUXDS:
  case LIBXSMM_PPC64LE_INSTR_XVCVDPSXDS:
  case LIBXSMM_PPC64LE_INSTR_XVCVUXDDP:
  case LIBXSMM_PPC64LE_INSTR_XVCVSXDDP:
  case LIBXSMM_PPC64LE_INSTR_XSRDPI:
  case LIBXSMM_PPC64LE_INSTR_XSRDPIZ:
  case LIBXSMM_PPC64LE_INSTR_XSRDPIP:
  case LIBXSMM_PPC64LE_INSTR_XSRDPIM:
  case LIBXSMM_PPC64LE_INSTR_XVRSPI:
  case LIBXSMM_PPC64LE_INSTR_XVRSPIZ:
  case LIBXSMM_PPC64LE_INSTR_XVRSPIP:
  case LIBXSMM_PPC64LE_INSTR_XVRSPIM:
  case LIBXSMM_PPC64LE_INSTR_XVRDPI:
  case LIBXSMM_PPC64LE_INSTR_XVRDPIZ:
  case LIBXSMM_PPC64LE_INSTR_XVRDPIP:
  case LIBXSMM_PPC64LE_INSTR_XVRDPIM:
  case LIBXSMM_PPC64LE_INSTR_XSCVDPSP:
  case LIBXSMM_PPC64LE_INSTR_XSRSP:
  case LIBXSMM_PPC64LE_INSTR_XSCVSPDP:
  case LIBXSMM_PPC64LE_INSTR_XSABSDP:
  case LIBXSMM_PPC64LE_INSTR_XSNABSDP:
  case LIBXSMM_PPC64LE_INSTR_XSNEGDP:
  case LIBXSMM_PPC64LE_INSTR_XVCVDPSP:
  case LIBXSMM_PPC64LE_INSTR_XVABSSP:
  case LIBXSMM_PPC64LE_INSTR_XVNABSSP:
  case LIBXSMM_PPC64LE_INSTR_XVNEGSP:
  case LIBXSMM_PPC64LE_INSTR_XVCVSPDP:
  case LIBXSMM_PPC64LE_INSTR_XVABSDP:
  case LIBXSMM_PPC64LE_INSTR_XVNABSDP:
  case LIBXSMM_PPC64LE_INSTR_XVNEGDP:
  case LIBXSMM_PPC64LE_INSTR_XSRSQRTESP:
  case LIBXSMM_PPC64LE_INSTR_XSRESP:
  case LIBXSMM_PPC64LE_INSTR_XSRSQRTEDP:
  case LIBXSMM_PPC64LE_INSTR_XSREDP:
  case LIBXSMM_PPC64LE_INSTR_XSSQRTSP:
  case LIBXSMM_PPC64LE_INSTR_XSSQRTDP:
  case LIBXSMM_PPC64LE_INSTR_XSRDPIC:
  case LIBXSMM_PPC64LE_INSTR_XVSQRTSP:
  case LIBXSMM_PPC64LE_INSTR_XVRSPIC:
  case LIBXSMM_PPC64LE_INSTR_XVSQRTDP:
  case LIBXSMM_PPC64LE_INSTR_XVRDPIC:
  case LIBXSMM_PPC64LE_INSTR_XSCVDPSPN:
  case LIBXSMM_PPC64LE_INSTR_XSCVSPDPN:
  case LIBXSMM_PPC64LE_INSTR_XSCVHPDP:
  case LIBXSMM_PPC64LE_INSTR_XSCVDPHP:
  case LIBXSMM_PPC64LE_INSTR_XVXEXPDP:
  case LIBXSMM_PPC64LE_INSTR_XVXEXPSP:
  case LIBXSMM_PPC64LE_INSTR_XVXSIGDP:
  case LIBXSMM_PPC64LE_INSTR_XVXSIGSP:
  case LIBXSMM_PPC64LE_INSTR_XXBRH:
  case LIBXSMM_PPC64LE_INSTR_XXBRW:
  case LIBXSMM_PPC64LE_INSTR_XXBRD:
  case LIBXSMM_PPC64LE_INSTR_XXBRQ:
  case LIBXSMM_PPC64LE_INSTR_XVCVBF16SP:
  case LIBXSMM_PPC64LE_INSTR_XVCVSPBF16:
  case LIBXSMM_PPC64LE_INSTR_XVCVHPSP:
  case LIBXSMM_PPC64LE_INSTR_XVCVSPHP:
    return libxsmm_ppc64le_xx2_form_3( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4 );
  default:
    LIBXSMM_HANDLE_ERROR( i_instr, LIBXSMM_ERR_UNKNOWN_OPERATION );
  }

}



LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_5_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5 ) {
  switch( i_instr ) {
  /* B (al) form */
  case LIBXSMM_PPC64LE_INSTR_UNDEF:
    return libxsmm_ppc64le_b_form_al( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5 );
  /* XX2 (4) form */
  case LIBXSMM_PPC64LE_INSTR_XXEXTRACTUW:
  case LIBXSMM_PPC64LE_INSTR_XXINSERTW:
    return libxsmm_ppc64le_xx2_form_4( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5 );
  /* XX3 (0) form */
  case LIBXSMM_PPC64LE_INSTR_XSCMPUDP:
  case LIBXSMM_PPC64LE_INSTR_XSCMPODP:
  case LIBXSMM_PPC64LE_INSTR_XSCMPEXPDP:
  case LIBXSMM_PPC64LE_INSTR_XVI8GER4:
  case LIBXSMM_PPC64LE_INSTR_XVI8GER4PP:
  case LIBXSMM_PPC64LE_INSTR_XVI8GER4SPP:
  case LIBXSMM_PPC64LE_INSTR_XVI4GER8:
  case LIBXSMM_PPC64LE_INSTR_XVI4GER8PP:
  case LIBXSMM_PPC64LE_INSTR_XVI16GER2:
  case LIBXSMM_PPC64LE_INSTR_XVI16GER2PP:
  case LIBXSMM_PPC64LE_INSTR_XVBF16GER2:
  case LIBXSMM_PPC64LE_INSTR_XVI16GER2S:
  case LIBXSMM_PPC64LE_INSTR_XVI16GER2SPP:
  case LIBXSMM_PPC64LE_INSTR_XVBF16GER2NN:
  case LIBXSMM_PPC64LE_INSTR_XVBF16GER2NP:
  case LIBXSMM_PPC64LE_INSTR_XVBF16GER2PN:
  case LIBXSMM_PPC64LE_INSTR_XVBF16GER2PP:
  case LIBXSMM_PPC64LE_INSTR_XVF16GER2:
  case LIBXSMM_PPC64LE_INSTR_XVF16GER2NN:
  case LIBXSMM_PPC64LE_INSTR_XVF16GER2NP:
  case LIBXSMM_PPC64LE_INSTR_XVF16GER2PN:
  case LIBXSMM_PPC64LE_INSTR_XVF16GER2PP:
  case LIBXSMM_PPC64LE_INSTR_XVF32GER:
  case LIBXSMM_PPC64LE_INSTR_XVF32GERNN:
  case LIBXSMM_PPC64LE_INSTR_XVF32GERNP:
  case LIBXSMM_PPC64LE_INSTR_XVF32GERPN:
  case LIBXSMM_PPC64LE_INSTR_XVF32GERPP:
  case LIBXSMM_PPC64LE_INSTR_XVF64GER:
  case LIBXSMM_PPC64LE_INSTR_XVF64GERNN:
  case LIBXSMM_PPC64LE_INSTR_XVF64GERNP:
  case LIBXSMM_PPC64LE_INSTR_XVF64GERPN:
  case LIBXSMM_PPC64LE_INSTR_XVF64GERPP:
    return libxsmm_ppc64le_xx3_form_0( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5 );
  default:
    LIBXSMM_HANDLE_ERROR( i_instr, LIBXSMM_ERR_UNKNOWN_OPERATION );
  }

}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_6_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5,
                                              unsigned int i_6 ) {
  switch( i_instr ) {
  /* MD form */
  case LIBXSMM_PPC64LE_INSTR_RLDICR:
    return libxsmm_ppc64le_md_form( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6 );
  /* XX3 (6) form */
  case LIBXSMM_PPC64LE_INSTR_XSMAXCDP:
  case LIBXSMM_PPC64LE_INSTR_XSMINCDP:
  case LIBXSMM_PPC64LE_INSTR_XSMAXJDP:
  case LIBXSMM_PPC64LE_INSTR_XSMINJDP:
  case LIBXSMM_PPC64LE_INSTR_XSMAXDP:
  case LIBXSMM_PPC64LE_INSTR_XSMINDP:
  case LIBXSMM_PPC64LE_INSTR_XSCPSGNDP:
  case LIBXSMM_PPC64LE_INSTR_XVMAXDP:
  case LIBXSMM_PPC64LE_INSTR_XVMAXSP:
  case LIBXSMM_PPC64LE_INSTR_XVMINDP:
  case LIBXSMM_PPC64LE_INSTR_XVMINSP:
  case LIBXSMM_PPC64LE_INSTR_XVIEXPDP:
  case LIBXSMM_PPC64LE_INSTR_XVIEXPSP:
  case LIBXSMM_PPC64LE_INSTR_XVCPSGNDP:
  case LIBXSMM_PPC64LE_INSTR_XVCPSGNSP:
  case LIBXSMM_PPC64LE_INSTR_XSMADDASP:
  case LIBXSMM_PPC64LE_INSTR_XSMADDMSP:
  case LIBXSMM_PPC64LE_INSTR_XSMADDADP:
  case LIBXSMM_PPC64LE_INSTR_XSMADDMDP:
  case LIBXSMM_PPC64LE_INSTR_XSMSUBASP:
  case LIBXSMM_PPC64LE_INSTR_XSMSUBMSP:
  case LIBXSMM_PPC64LE_INSTR_XSMSUBADP:
  case LIBXSMM_PPC64LE_INSTR_XSMSUBMDP:
  case LIBXSMM_PPC64LE_INSTR_XVMADDASP:
  case LIBXSMM_PPC64LE_INSTR_XVMADDMSP:
  case LIBXSMM_PPC64LE_INSTR_XVMADDADP:
  case LIBXSMM_PPC64LE_INSTR_XVMADDMDP:
  case LIBXSMM_PPC64LE_INSTR_XVMSUBASP:
  case LIBXSMM_PPC64LE_INSTR_XVMSUBMSP:
  case LIBXSMM_PPC64LE_INSTR_XVMSUBADP:
  case LIBXSMM_PPC64LE_INSTR_XVMSUBMDP:
  case LIBXSMM_PPC64LE_INSTR_XSNMADDASP:
  case LIBXSMM_PPC64LE_INSTR_XSNMADDMSP:
  case LIBXSMM_PPC64LE_INSTR_XSNMADDADP:
  case LIBXSMM_PPC64LE_INSTR_XSNMADDMDP:
  case LIBXSMM_PPC64LE_INSTR_XSNMSUBASP:
  case LIBXSMM_PPC64LE_INSTR_XSNMSUBMSP:
  case LIBXSMM_PPC64LE_INSTR_XSNMSUBADP:
  case LIBXSMM_PPC64LE_INSTR_XSNMSUBMDP:
  case LIBXSMM_PPC64LE_INSTR_XVNMADDASP:
  case LIBXSMM_PPC64LE_INSTR_XVNMADDMSP:
  case LIBXSMM_PPC64LE_INSTR_XVNMADDADP:
  case LIBXSMM_PPC64LE_INSTR_XVNMADDMDP:
  case LIBXSMM_PPC64LE_INSTR_XVNMSUBASP:
  case LIBXSMM_PPC64LE_INSTR_XVNMSUBMSP:
  case LIBXSMM_PPC64LE_INSTR_XVNMSUBADP:
  case LIBXSMM_PPC64LE_INSTR_XVNMSUBMDP:
  case LIBXSMM_PPC64LE_INSTR_XXMRGHW:
  case LIBXSMM_PPC64LE_INSTR_XXMRGLW:
  case LIBXSMM_PPC64LE_INSTR_XXPERM:
  case LIBXSMM_PPC64LE_INSTR_XXPERMR:
  case LIBXSMM_PPC64LE_INSTR_XXLAND:
  case LIBXSMM_PPC64LE_INSTR_XXLANDC:
  case LIBXSMM_PPC64LE_INSTR_XXLNAND:
  case LIBXSMM_PPC64LE_INSTR_XXLEQV:
  case LIBXSMM_PPC64LE_INSTR_XXLNOR:
  case LIBXSMM_PPC64LE_INSTR_XXLORC:
  case LIBXSMM_PPC64LE_INSTR_XXLOR:
  case LIBXSMM_PPC64LE_INSTR_XXLXOR:
  case LIBXSMM_PPC64LE_INSTR_XSCMPEQDP:
  case LIBXSMM_PPC64LE_INSTR_XSCMPGTDP:
  case LIBXSMM_PPC64LE_INSTR_XSCMPGEDP:
    return libxsmm_ppc64le_xx3_form_6( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6 );
  default:
    LIBXSMM_HANDLE_ERROR( i_instr, LIBXSMM_ERR_UNKNOWN_OPERATION );
  }

}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_7_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5,
                                              unsigned int i_6,
                                              unsigned int i_7 ) {
  switch( i_instr ) {
  /* XX3 (3) form */
  case LIBXSMM_PPC64LE_INSTR_XVSLDWI:
  case LIBXSMM_PPC64LE_INSTR_XXPERMDI:
    return libxsmm_ppc64le_xx3_form_3( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7 );
  /* XX3 (5) form */
  case LIBXSMM_PPC64LE_INSTR_XVCMPEQDP:
  case LIBXSMM_PPC64LE_INSTR_XVCMPEQSP:
  case LIBXSMM_PPC64LE_INSTR_XVCMPGTDP:
  case LIBXSMM_PPC64LE_INSTR_XVCMPGTSP:
  case LIBXSMM_PPC64LE_INSTR_XVCMPGEDP:
  case LIBXSMM_PPC64LE_INSTR_XVCMPGESP:
    return libxsmm_ppc64le_xx3_form_5( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7 );
  default:
    LIBXSMM_HANDLE_ERROR( i_instr, LIBXSMM_ERR_UNKNOWN_OPERATION );
  }
}


LIBXSMM_API_INTERN
unsigned int libxsmm_ppc64le_instr_8_wrapper( unsigned int i_instr,
                                              unsigned int i_1,
                                              unsigned int i_2,
                                              unsigned int i_3,
                                              unsigned int i_4,
                                              unsigned int i_5,
                                              unsigned int i_6,
                                              unsigned int i_7,
                                              unsigned int i_8 ) {
  switch( i_instr ) {
  /* XX4 form */
  case LIBXSMM_PPC64LE_INSTR_XXSEL:
    return libxsmm_ppc64le_xx4_form_3( i_instr, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, (unsigned char)i_7, (unsigned char)i_8 );
  default:
    LIBXSMM_HANDLE_ERROR( i_instr, LIBXSMM_ERR_UNKNOWN_OPERATION );
  }
}


/* b form */
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

/* ds form */
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


/* d form */
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

/* */
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
